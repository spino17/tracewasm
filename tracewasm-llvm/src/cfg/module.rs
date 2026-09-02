//! The module: its target settings and the functions it defines.

use crate::{
    cfg::{function::FuncId, global::Global},
    interner::StrId,
    value::FuncSignature,
};
use rustc_hash::FxHashMap;
use std::fmt::Display;

/// A target triple: `arch-vendor-os` with an optional environment.
///
/// The fields are strings rather than enums because the sets are open-ended — new
/// architectures and operating systems appear, and rejecting an unknown one would
/// refuse a target LLVM supports.
///
/// ```
/// # use tracewasm_llvm::cfg::module::Triple;
/// let t = Triple::new("arm64".into(), "apple".into(), "macosx".into(), None);
/// assert_eq!(t.to_string(), "arm64-apple-macosx");
///
/// let gnu = Triple::new(
///     "x86_64".into(), "unknown".into(), "linux".into(), Some("gnu".into()),
/// );
/// assert_eq!(gnu.to_string(), "x86_64-unknown-linux-gnu");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triple {
    arch: String,
    vendor: String,
    os: String,
    env: Option<String>,
}

impl Triple {
    /// A triple from its parts.
    pub fn new(arch: String, vendor: String, os: String, env: Option<String>) -> Self {
        Triple {
            arch,
            vendor,
            os,
            env,
        }
    }
}

// `Display` rather than `ToString` directly: the blanket impl gives `to_string` for
// free, and implementing it by hand opts out of every formatting context.
impl Display for Triple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-{}", self.arch, self.vendor, self.os)?;

        if let Some(env) = &self.env {
            write!(f, "-{env}")?;
        }

        Ok(())
    }
}

/// Byte order, the `e`/`E` specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// `e` — least significant bits first.
    Little,
    /// `E` — most significant bits first.
    Big,
}

impl Display for Endianness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Endianness::Little => "e",
            Endianness::Big => "E",
        })
    }
}

/// How symbols are mangled, the `m:` specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mangling {
    /// `m:e` — ELF.
    Elf,
    /// `m:o` — Mach-O.
    MachO,
    /// `m:w` — Windows COFF.
    WindowsCoff,
    /// `m:x` — Windows COFF on x86.
    WindowsCoffX86,
    /// `m:a` — XCOFF.
    XCoff,
}

impl Display for Mangling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mangling::Elf => "m:e",
            Mangling::MachO => "m:o",
            Mangling::WindowsCoff => "m:w",
            Mangling::WindowsCoffX86 => "m:x",
            Mangling::XCoff => "m:a",
        })
    }
}

/// One entry of a [`DataLayout`].
///
/// Only the specifications this crate needs are modelled. An alignment pair is
/// `<abi>[:<preferred>]`, and omitting the preferred alignment lets it default to the
/// ABI one — which is how LLVM reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataLayoutSpec {
    /// Byte order.
    Endianness(Endianness),
    /// Symbol mangling.
    Mangling(Mangling),
    /// `p[n]:<size>:<abi>[:<pref>]` — pointer size and alignment, in bits.
    Pointer {
        /// Address space, or `None` for the default one.
        address_space: Option<u32>,
        /// Pointer size in bits.
        size: u32,
        /// ABI alignment in bits.
        abi: u32,
        /// Preferred alignment in bits.
        pref: Option<u32>,
    },
    /// `i<size>:<abi>[:<pref>]` — integer alignment.
    Int {
        /// Width in bits.
        size: u32,
        /// ABI alignment in bits.
        abi: u32,
        /// Preferred alignment in bits.
        pref: Option<u32>,
    },
    /// `f<size>:<abi>[:<pref>]` — float alignment.
    Float {
        /// Width in bits.
        size: u32,
        /// ABI alignment in bits.
        abi: u32,
        /// Preferred alignment in bits.
        pref: Option<u32>,
    },
    /// `a:<abi>[:<pref>]` — aggregate alignment.
    Aggregate {
        /// ABI alignment in bits.
        abi: u32,
        /// Preferred alignment in bits.
        pref: Option<u32>,
    },
    /// `n<w1>:<w2>:…` — the integer widths the target has registers for.
    NativeIntWidths(Vec<u32>),
    /// `S<n>` — natural stack alignment in bits.
    StackAlignment(u32),
}

impl Display for DataLayoutSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// `<abi>[:<pref>]`, shared by every alignment spec.
        fn alignment(
            f: &mut std::fmt::Formatter<'_>,
            abi: u32,
            pref: Option<u32>,
        ) -> std::fmt::Result {
            write!(f, "{abi}")?;

            if let Some(pref) = pref {
                write!(f, ":{pref}")?;
            }

            Ok(())
        }

        match self {
            DataLayoutSpec::Endianness(e) => write!(f, "{e}"),
            DataLayoutSpec::Mangling(m) => write!(f, "{m}"),
            DataLayoutSpec::Pointer {
                address_space,
                size,
                abi,
                pref,
            } => {
                f.write_str("p")?;

                if let Some(space) = address_space {
                    write!(f, "{space}")?;
                }

                write!(f, ":{size}:")?;
                alignment(f, *abi, *pref)
            }
            DataLayoutSpec::Int { size, abi, pref } => {
                write!(f, "i{size}:")?;
                alignment(f, *abi, *pref)
            }
            DataLayoutSpec::Float { size, abi, pref } => {
                write!(f, "f{size}:")?;
                alignment(f, *abi, *pref)
            }
            DataLayoutSpec::Aggregate { abi, pref } => {
                f.write_str("a:")?;
                alignment(f, *abi, *pref)
            }
            DataLayoutSpec::NativeIntWidths(widths) => {
                f.write_str("n")?;

                for (i, width) in widths.iter().enumerate() {
                    if i != 0 {
                        f.write_str(":")?;
                    }

                    write!(f, "{width}")?;
                }

                Ok(())
            }
            DataLayoutSpec::StackAlignment(bits) => write!(f, "S{bits}"),
        }
    }
}

/// A target's data layout: sizes, alignments and byte order.
///
/// LLVM validates this string — `target datalayout = "not-a-layout"` is refused with
/// "size must be a non-zero 24-bit integer" — so building it from typed
/// [`DataLayoutSpec`]s rather than free text makes a malformed layout unrepresentable.
///
/// [`Default`] is an empty layout, meaning *unset*: the emitter omits the
/// `target datalayout` line entirely rather than writing an empty one.
///
/// ```
/// # use tracewasm_llvm::cfg::module::{DataLayout, DataLayoutSpec, Endianness, Mangling};
/// let layout = DataLayout::new(vec![
///     DataLayoutSpec::Endianness(Endianness::Little),
///     DataLayoutSpec::Mangling(Mangling::MachO),
///     DataLayoutSpec::Int { size: 64, abi: 64, pref: None },
///     DataLayoutSpec::NativeIntWidths(vec![32, 64]),
///     DataLayoutSpec::StackAlignment(128),
/// ]);
///
/// assert_eq!(layout.to_string(), "e-m:o-i64:64-n32:64-S128");
/// assert_eq!(DataLayout::default().to_string(), "");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DataLayout {
    specs: Vec<DataLayoutSpec>,
}

impl DataLayout {
    /// A layout from its specifications, rendered in the order given.
    pub fn new(specs: Vec<DataLayoutSpec>) -> Self {
        DataLayout { specs }
    }
}

impl Display for DataLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, spec) in self.specs.iter().enumerate() {
            if i != 0 {
                f.write_str("-")?;
            }

            write!(f, "{spec}")?;
        }

        Ok(())
    }
}

/// One LLVM module: target settings, globals and functions.
///
/// Owned by the [`Context`](crate::cfg::context::Context), which the finished
/// [`ControlFlowGraph`](crate::cfg::ControlFlowGraph) takes over. Functions are held
/// as ids into the context's arena; `func_names` maps each name to its signature,
/// which is what makes a duplicate `@name` a build error rather than something
/// `llvm-as` discovers later.
///
/// The target strings are rendered from a [`Triple`] and a [`DataLayout`] at
/// construction. An empty `data_layout` means "unset" and the emitter omits the
/// line; a structured [`Triple`] is always present, so the triple line always
/// appears.
pub struct Module {
    pub(crate) triple: String,
    pub(crate) data_layout: String,
    pub(crate) globals: Vec<Global>,
    pub(crate) functions: Vec<FuncId>,
    pub(crate) func_names: FxHashMap<StrId, FuncSignature>,
}

impl Module {
    /// An empty module for the given target.
    pub(crate) fn new(triple: Triple, data_layout: DataLayout) -> Self {
        Module {
            triple: triple.to_string(),
            data_layout: data_layout.to_string(),
            globals: vec![],
            functions: vec![],
            func_names: FxHashMap::default(),
        }
    }
}
