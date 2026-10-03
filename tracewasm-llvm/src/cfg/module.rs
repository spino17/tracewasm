//! The module: its target settings and the functions it defines.

use crate::{
    cfg::{function::FuncId, global::GlobalData},
    error::TargetParseError,
    interner::StrId,
};
use rustc_hash::FxHashMap;
use std::{fmt::Display, str::FromStr};

/// What a module says about the machine its IR is for.
///
/// Chosen once, when the [`Context`](crate::cfg::context::Context) is created, so
/// every layout-dependent decision made while building sees the same target. A data
/// layout without a triple can't be expressed.
///
/// ```
/// # use tracewasm_llvm::cfg::{context::Context, module::{Target, Triple}};
/// // Let whatever consumes the IR decide: the crate's JIT uses the host's.
/// let for_the_jit = Context::new(Target::Unspecified);
///
/// // Or pin the platform, leaving its layout to the consumer.
/// let for_apple_silicon = Context::new(Target::Triple(
///     Triple::new("arm64".into(), "apple".into(), "macosx".into(), None),
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// No `target` lines at all; the consumer decides. `llvm-as` and `opt` accept
    /// that, and the crate's JIT supplies the host's triple and layout.
    Unspecified,
    /// A `target triple`, with the data layout left to the consumer.
    Triple(Triple),
    /// Both, for IR that has to mean exactly one machine. With the `jit` feature,
    /// `JITHandler::target` returns the host's in this form.
    Full {
        /// The platform the IR is written for.
        triple: Triple,
        /// Its sizes, alignments and byte order.
        data_layout: DataLayout,
    },
}

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

/// Parses `arch-vendor-os[-env]`, the inverse of the `Display` impl: anything past
/// the third `-` is the environment, so rendering the result gives back the input.
///
/// ```
/// # use tracewasm_llvm::cfg::module::Triple;
/// let t: Triple = "arm64-apple-darwin25.1.0".parse().unwrap();
/// assert_eq!(t.to_string(), "arm64-apple-darwin25.1.0");
/// assert!("x86_64-linux".parse::<Triple>().is_err());
/// ```
impl FromStr for Triple {
    type Err = TargetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || TargetParseError::Triple(s.to_string());
        let mut parts = s.splitn(4, '-');
        let mut next = || parts.next().filter(|p| !p.is_empty()).ok_or_else(err);

        let arch = next()?.to_string();
        let vendor = next()?.to_string();
        let os = next()?.to_string();
        let env = match parts.next() {
            Some("") => return Err(err()),
            env => env.map(str::to_string),
        };

        Ok(Triple::new(arch, vendor, os, env))
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
    /// `v<size>:<abi>[:<pref>]` — vector alignment.
    Vector {
        /// Width in bits.
        size: u32,
        /// ABI alignment in bits.
        abi: u32,
        /// Preferred alignment in bits.
        pref: Option<u32>,
    },
    /// `S<n>` — natural stack alignment in bits.
    StackAlignment(u32),
    /// `F<type><abi>` — function pointer alignment, in bits.
    FunctionPointerAlignment {
        /// `false` for `Fi` (independent of the functions' own alignment), `true`
        /// for `Fn` (a multiple of it).
        multiple_of_function_alignment: bool,
        /// ABI alignment in bits.
        abi: u32,
    },
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
            DataLayoutSpec::Vector { size, abi, pref } => {
                write!(f, "v{size}:")?;
                alignment(f, *abi, *pref)
            }
            DataLayoutSpec::StackAlignment(bits) => write!(f, "S{bits}"),
            DataLayoutSpec::FunctionPointerAlignment {
                multiple_of_function_alignment,
                abi,
            } => write!(
                f,
                "F{}{abi}",
                if *multiple_of_function_alignment {
                    "n"
                } else {
                    "i"
                }
            ),
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

/// Parses an LLVM data layout string, the inverse of the `Display` impl: every
/// specification is kept, in order, so rendering the result gives back the input
/// exactly. The empty string is the unset layout.
///
/// A specification this crate doesn't model (`A`, `P`, `G`, `ni:`, a pointer's index
/// width, …) is an error rather than something to skip, because a layout missing it
/// describes a different machine.
///
/// ```
/// # use tracewasm_llvm::cfg::module::DataLayout;
/// let host = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32";
/// let layout: DataLayout = host.parse().unwrap();
///
/// assert_eq!(layout.to_string(), host);
/// assert!("e-A5".parse::<DataLayout>().is_err());
/// ```
impl FromStr for DataLayout {
    type Err = TargetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Ok(DataLayout::default());
        }

        s.split('-')
            .map(parse_spec)
            .collect::<Result<_, _>>()
            .map(DataLayout::new)
    }
}

/// One `-`-separated specification of a data layout string.
fn parse_spec(spec: &str) -> Result<DataLayoutSpec, TargetParseError> {
    let err = || TargetParseError::DataLayoutSpec(spec.to_string());
    let num = |n: &str| n.parse::<u32>().map_err(|_| err());

    // `<abi>[:<pref>]`, the tail every alignment spec shares.
    let alignment = |fields: &[&str]| -> Result<(u32, Option<u32>), TargetParseError> {
        match fields {
            [abi] => Ok((num(abi)?, None)),
            [abi, pref] => Ok((num(abi)?, Some(num(pref)?))),
            _ => Err(err()),
        }
    };

    let mut chars = spec.chars();
    let (kind, rest) = match chars.next() {
        Some(kind) => (kind, chars.as_str()),
        None => return Err(err()),
    };
    let fields: Vec<&str> = rest.split(':').collect();

    Ok(match kind {
        'e' if rest.is_empty() => DataLayoutSpec::Endianness(Endianness::Little),
        'E' if rest.is_empty() => DataLayoutSpec::Endianness(Endianness::Big),
        'm' => DataLayoutSpec::Mangling(match rest {
            ":e" => Mangling::Elf,
            ":o" => Mangling::MachO,
            ":w" => Mangling::WindowsCoff,
            ":x" => Mangling::WindowsCoffX86,
            ":a" => Mangling::XCoff,
            _ => return Err(err()),
        }),
        'p' => {
            let [space, size, tail @ ..] = fields.as_slice() else {
                return Err(err());
            };
            let (abi, pref) = alignment(tail)?;

            DataLayoutSpec::Pointer {
                address_space: if space.is_empty() {
                    None
                } else {
                    Some(num(space)?)
                },
                size: num(size)?,
                abi,
                pref,
            }
        }
        'i' | 'f' | 'v' => {
            let [size, tail @ ..] = fields.as_slice() else {
                return Err(err());
            };
            let size = num(size)?;
            let (abi, pref) = alignment(tail)?;

            match kind {
                'i' => DataLayoutSpec::Int { size, abi, pref },
                'f' => DataLayoutSpec::Float { size, abi, pref },
                _ => DataLayoutSpec::Vector { size, abi, pref },
            }
        }
        'a' => {
            let ["", tail @ ..] = fields.as_slice() else {
                return Err(err());
            };
            let (abi, pref) = alignment(tail)?;

            DataLayoutSpec::Aggregate { abi, pref }
        }
        'n' => DataLayoutSpec::NativeIntWidths(
            fields.iter().map(|w| num(w)).collect::<Result<_, _>>()?,
        ),
        'S' => DataLayoutSpec::StackAlignment(num(rest)?),
        'F' => {
            let (multiple_of_function_alignment, abi) = if let Some(abi) = rest.strip_prefix('n') {
                (true, abi)
            } else if let Some(abi) = rest.strip_prefix('i') {
                (false, abi)
            } else {
                return Err(err());
            };

            DataLayoutSpec::FunctionPointerAlignment {
                multiple_of_function_alignment,
                abi: num(abi)?,
            }
        }
        _ => return Err(err()),
    })
}

/// One LLVM module: target settings, globals and functions.
///
/// Owned by the [`Context`](crate::cfg::context::Context), which the finished
/// [`ControlFlowGraph`](crate::cfg::ControlFlowGraph) takes over. Functions are held
/// as ids into the context's arena; `globals` maps each `@name`, function or
/// variable, to its [`GlobalData`], which is what makes a duplicate `@name` a build error rather than something
/// `llvm-as` discovers later.
///
/// The target strings are rendered from the [`Target`] the context was created
/// with. Whatever it leaves unset is the empty string, and the emitter writes no
/// line for it.
pub struct Module {
    pub(crate) triple: String,
    pub(crate) data_layout: String,
    pub(crate) functions: Vec<FuncId>,
    pub(crate) imported_functions: Vec<StrId>,
    pub(crate) global_variables: Vec<StrId>,
    pub(crate) globals: FxHashMap<StrId, GlobalData>,
}

impl Module {
    /// An empty module for the given target.
    pub(crate) fn new(target: Target) -> Self {
        let (triple, data_layout) = match target {
            Target::Unspecified => (String::new(), String::new()),
            Target::Triple(triple) => (triple.to_string(), String::new()),
            Target::Full {
                triple,
                data_layout,
            } => (triple.to_string(), data_layout.to_string()),
        };

        Module {
            triple,
            data_layout,
            functions: vec![],
            imported_functions: vec![],
            global_variables: vec![],
            globals: FxHashMap::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parsing and rendering have to be exact inverses: the JIT compares data
    /// layouts as strings, so a round trip that changed a single character would
    /// make a module built for the host fail to match it.
    #[test]
    fn real_data_layouts_round_trip_exactly() {
        for layout in [
            // arm64 macOS and x86-64 Linux, as LLVM 22 reports them.
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32",
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
            // Big-endian, a preferred alignment, a vector, an aggregate, `Fi`.
            "E-m:e-p:32:32:64-i1:8:32-f64:64-v128:128:128-a:0:64-Fi8",
            "",
        ] {
            let parsed: DataLayout = layout.parse().unwrap();

            assert_eq!(parsed.to_string(), layout);
        }
    }

    #[test]
    fn each_spec_parses_to_what_it_names() {
        let layout: DataLayout = "e-m:o-p270:32:32-i64:64:128-v128:128-a:0:64-n32:64-S128-Fn32"
            .parse()
            .unwrap();

        assert_eq!(
            layout,
            DataLayout::new(vec![
                DataLayoutSpec::Endianness(Endianness::Little),
                DataLayoutSpec::Mangling(Mangling::MachO),
                DataLayoutSpec::Pointer {
                    address_space: Some(270),
                    size: 32,
                    abi: 32,
                    pref: None
                },
                DataLayoutSpec::Int {
                    size: 64,
                    abi: 64,
                    pref: Some(128)
                },
                DataLayoutSpec::Vector {
                    size: 128,
                    abi: 128,
                    pref: None
                },
                DataLayoutSpec::Aggregate {
                    abi: 0,
                    pref: Some(64)
                },
                DataLayoutSpec::NativeIntWidths(vec![32, 64]),
                DataLayoutSpec::StackAlignment(128),
                DataLayoutSpec::FunctionPointerAlignment {
                    multiple_of_function_alignment: true,
                    abi: 32
                },
            ])
        );
    }

    /// A specification this crate doesn't model is refused rather than skipped:
    /// dropping it would describe a different machine.
    #[test]
    fn unmodelled_or_malformed_specs_are_refused() {
        for (layout, bad) in [
            ("e-A5", "A5"),                       // alloca address space
            ("e-P1", "P1"),                       // program address space
            ("e-ni:10", "ni:10"),                 // non-integral pointers
            ("e-p:64:64:64:32", "p:64:64:64:32"), // a pointer's index width
            ("e-m:q", "m:q"),                     // unknown mangling
            ("e-i64", "i64"),                     // no alignment
            ("e-ix:64", "ix:64"),                 // not a number
            ("e-Fx8", "Fx8"),                     // neither `Fi` nor `Fn`
            ("e--S128", ""),                      // an empty spec
        ] {
            assert_eq!(
                layout.parse::<DataLayout>(),
                Err(TargetParseError::DataLayoutSpec(bad.to_string())),
                "{layout}"
            );
        }
    }

    #[test]
    fn triples_round_trip_exactly() {
        for triple in [
            "arm64-apple-macosx",
            "arm64-apple-darwin25.1.0",
            "x86_64-unknown-linux-gnu",
            "armv7-unknown-linux-gnueabihf",
            "x86_64-pc-windows-msvc-coff",
        ] {
            let parsed: Triple = triple.parse().unwrap();

            assert_eq!(parsed.to_string(), triple);
        }
    }

    #[test]
    fn a_triple_needs_arch_vendor_and_os() {
        for triple in [
            "",
            "x86_64",
            "x86_64-linux",
            "x86_64--linux",
            "x86_64-pc-linux-",
        ] {
            assert_eq!(
                triple.parse::<Triple>(),
                Err(TargetParseError::Triple(triple.to_string())),
                "{triple:?}"
            );
        }
    }
}
