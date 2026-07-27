use gimli::{Dwarf, EndianSlice, RunTimeEndian, SectionId};
use rustc_demangle::demangle;
use std::{borrow::Cow, fs};
use tracewasm_core::module::Module;
use tracewasm_macros::imports;

/// Example host module. The `#[imports]` macro reads the `#[module("...")]`-tagged
/// (and any `#[global("...")]`-tagged) methods below and generates the entire
/// [`ImportRegistry`] impl (`execute`, `signature`, `func_count`, `global_count`,
/// `get_global`) — the embedder writes only the function bodies.
///
/// Serves as the example host state / import registry for the playground.
pub struct ImportedFunctions {
    /// Call counter mutated by the host functions to demonstrate `&mut self` state.
    count: u32,
}

#[imports]
impl ImportedFunctions {
    #[module("env")]
    fn host1(&mut self, a: i32, b: i32) -> (i32,) {
        self.count += 1;
        (a.wrapping_add(b),)
    }

    #[module("env")]
    fn host2(&mut self, a: i32, b: i32, c: i64) -> (i32,) {
        (a.wrapping_add(b).wrapping_add(c as i32),)
    }
}

fn main() -> Result<(), anyhow::Error> {
    let buf = fs::read(
        "/Users/bhavyabhatt/Desktop/bhavya/projects/tracewasm/target/wasm32-unknown-unknown/debug/tracewasm_scratch.wasm",
    )?;

    struct DebugSection {
        debug_line: Box<[u8]>,
        debug_loc: Box<[u8]>,
        debug_abbrev: Box<[u8]>,
        debug_ranges: Box<[u8]>,
        debug_info: Box<[u8]>,
        debug_str: Box<[u8]>,
        target_features: Box<[u8]>,
    }

    let module = Module::compile(&buf)?;
    let custom_unknowns = &module.custom_section.unknown;

    let mut debug_line: Option<Box<[u8]>> = None;
    let mut debug_loc: Option<Box<[u8]>> = None;
    let mut debug_abbrev: Option<Box<[u8]>> = None;
    let mut debug_ranges: Option<Box<[u8]>> = None;
    let mut debug_info: Option<Box<[u8]>> = None;
    let mut debug_str: Option<Box<[u8]>> = None;

    for (key, value) in custom_unknowns {
        match key.as_ref() {
            ".debug_line" => debug_line = Some(value.clone()),
            ".debug_loc" => debug_loc = Some(value.clone()),
            ".debug_abbrev" => debug_abbrev = Some(value.clone()),
            ".debug_ranges" => debug_ranges = Some(value.clone()),
            ".debug_info" => debug_info = Some(value.clone()),
            ".debug_str" => debug_str = Some(value.clone()),
            _ => continue,
        }
    }

    debug_assert!(debug_line.is_some());
    debug_assert!(debug_abbrev.is_some());
    debug_assert!(debug_info.is_some());
    debug_assert!(debug_loc.is_some());
    debug_assert!(debug_ranges.is_some());
    debug_assert!(debug_str.is_some());

    let dwarf_cow = Dwarf::load(|id: SectionId| -> Result<Cow<[u8]>, gimli::Error> {
        Ok(match custom_unknowns.get(id.name()) {
            Some(v) => Cow::Borrowed(v),
            None => Cow::Borrowed(&[][..]),
        })
    })?;

    let dwarf = dwarf_cow.borrow(|s| EndianSlice::new(s, RunTimeEndian::Little));

    println!("{:?}", dwarf);

    let mut rows: Vec<(u64, String, u64)> = Vec::new(); // (code_offset, file, line)

    let mut units = dwarf.units();

    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;

        let Some(program) = unit.line_program.clone() else {
            continue;
        };

        let mut sm = program.rows();

        while let Some((hdr, row)) = sm.next_row()? {
            if row.end_sequence() {
                continue;
            }

            let line = row.line().map(|l| l.get()).unwrap_or(0);

            let file = row
                .file(hdr)
                .map(|f| {
                    let dir = f
                        .directory(hdr)
                        .and_then(|d| dwarf.attr_string(&unit, d).ok())
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let name = dwarf
                        .attr_string(&unit, f.path_name())
                        .ok()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    format!("{dir}/{name}") // optionally prepend unit.comp_dir too
                })
                .unwrap_or_default();

            rows.push((row.address(), file, line)); // address = code-section offset
        }
    }

    rows.sort_by_key(|r| r.0);

    println!("rows: {:?}", rows);

    let registry = ImportedFunctions { count: 0 };

    let func = module.get_typed_func::<(i32, i64), (i64,)>("bench_bits")?;

    /*let mut instance = module.instantiate::<LinearMemory, _>(registry, None)?;
    let res = func.call((1, 2), &mut instance)?.0;*/

    let s = "$_RNvNtNtCs3O6bguQwcd4_4core9panicking11panic_const24panic_const_div_overflow";
    let s = s.strip_prefix('$').unwrap_or(s);
    let str = demangle(s);
    // println!("{:#}", str); // core::panicking::panic_const::panic_const_div_overflow

    Ok(())
}
