use crate::{
    cfg::{
        basic_block::BasicBlock,
        function::{FuncId, Function},
    },
    error::BuildError,
    interner::{ConstInterner, StrId, StrInterner},
};
use id_arena::Arena;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;

#[derive(Default)]
pub struct Context {
    pub(crate) blocks: Arena<BasicBlock>,
    pub(crate) funcs: Arena<Function>,
    pub(crate) str_interner: StrInterner,
    pub(crate) const_interner: ConstInterner,
    pub(crate) reg_name_assigner: FxHashMap<FuncId, FuncRegNameIndex>,
    pub(crate) register_def_instr_index: FxHashMap<FuncId, FxHashMap<StrId, usize>>,
}

impl Context {
    pub(crate) fn name_for_reg(
        &mut self,
        name: Option<&str>,
        func_id: FuncId,
    ) -> Result<String, BuildError> {
        let assigner = self.reg_name_assigner.entry(func_id).or_default();
        let name = assigner.name_from_hint(name)?;

        Ok(name)
    }
}

#[derive(Default)]
struct FuncRegNameIndex {
    unnamed_index: u32,
    named_index: FxHashMap<String, u32>,
    issued_names: FxHashSet<String>,
}

impl FuncRegNameIndex {
    fn next_unnamed_index(&mut self) -> u32 {
        let index = self.unnamed_index;

        self.unnamed_index += 1;

        index
    }

    fn next_named_index(&mut self, name: &str) -> u32 {
        match self.named_index.entry(name.to_string()) {
            Entry::Occupied(mut occ) => {
                let index = occ.get_mut();
                let id = *index;

                *index += 1;

                id
            }
            Entry::Vacant(vac) => {
                vac.insert(1);

                0
            }
        }
    }

    fn name_from_hint(&mut self, hint: Option<&str>) -> Result<String, BuildError> {
        let Some(hint) = hint else {
            return Ok(self.next_unnamed_index().to_string());
        };

        let re = Regex::new(r"^[-a-zA-Z$._][-a-zA-Z$._0-9]*$").unwrap();

        if !re.is_match(hint) {
            return Err(BuildError::InvalidRegisterName(hint.to_string()));
        }

        let final_name = loop {
            let index = self.next_named_index(hint);

            let name = if index == 0 {
                hint.to_string()
            } else {
                format!("{}{}", hint, index)
            };

            if !self.issued_names.contains(&name) {
                break name;
            }
        };

        self.issued_names.insert(final_name.clone());

        Ok(final_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cfg::Builder, test_support::fixture, value::Type};

    /// A context, a builder, and two functions to scope names against.
    fn two_functions() -> (Context, FuncId, FuncId) {
        let mut ctx = Context::default();
        let mut builder = Builder::new(String::new(), String::new());

        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let g = builder.add_function("g".to_string(), &mut ctx).unwrap();

        (ctx, f, g)
    }

    fn name(ctx: &mut Context, hint: Option<&str>, func: FuncId) -> String {
        ctx.name_for_reg(hint, func)
            .unwrap_or_else(|e| panic!("hint {hint:?} should be accepted: {e}"))
    }

    /// LLVM numbers unnamed temporaries from 0, in order, and the numbering is per
    /// function — `%0` is the first unnamed value in *that* function's body.
    #[test]
    fn unnamed_values_are_numbered_from_zero_per_function() {
        let (mut ctx, f, g) = two_functions();

        let in_f: Vec<String> = (0..3).map(|_| name(&mut ctx, None, f)).collect();
        let in_g: Vec<String> = (0..2).map(|_| name(&mut ctx, None, g)).collect();

        assert_eq!(in_f, ["0", "1", "2"]);
        assert_eq!(in_g, ["0", "1"], "a second function restarts at 0");
    }

    /// Local names are scoped to their function, so the same hint in two functions
    /// is not a collision and neither needs a suffix.
    #[test]
    fn a_hint_is_scoped_to_its_function() {
        let (mut ctx, f, g) = two_functions();

        assert_eq!(name(&mut ctx, Some("sum"), f), "sum");
        assert_eq!(
            name(&mut ctx, Some("sum"), g),
            "sum",
            "another function's `sum` is a different value"
        );
    }

    /// Within one function a name may be defined once, so a repeated hint has to
    /// come back changed.
    #[test]
    fn a_repeated_hint_is_made_unique() {
        let (mut ctx, f, _) = two_functions();

        let names: Vec<String> = (0..3).map(|_| name(&mut ctx, Some("x"), f)).collect();

        assert_eq!(names, ["x", "x1", "x2"]);
    }

    /// The property all of the above serve: **every name a function hands out is
    /// distinct**. LLVM rejects a body with two definitions of the same local.
    ///
    /// This is the one that does not care *how* uniqueness is achieved, so it stays
    /// true if the suffixing scheme changes.
    #[test]
    fn every_name_in_one_function_is_distinct() {
        let (mut ctx, f, _) = two_functions();
        let mut issued = FxHashSet::default();

        // A hint, the same hint again, and a hint that looks like the suffixed form
        // of the first — plus unnamed values interleaved.
        let requests = [
            Some("x"),
            Some("x"),
            Some("x1"),
            None,
            Some("x"),
            None,
            Some("x2"),
        ];

        for hint in requests {
            let issued_name = name(&mut ctx, hint, f);

            assert!(
                issued.insert(issued_name.clone()),
                "`{issued_name}` was handed out twice (hint {hint:?}); LLVM rejects \
                 two definitions of the same local"
            );
        }
    }

    /// A hint that already looks like a suffixed name must not collide with the
    /// suffix the assigner would generate.
    #[test]
    fn a_hint_matching_a_generated_suffix_does_not_collide() {
        let (mut ctx, f, _) = two_functions();

        let first = name(&mut ctx, Some("x"), f);
        let suffixed = name(&mut ctx, Some("x"), f);
        let asked_for = name(&mut ctx, Some("x1"), f);

        assert_eq!((first.as_str(), suffixed.as_str()), ("x", "x1"));
        assert_ne!(
            suffixed, asked_for,
            "the generated `x1` and a requested `x1` are two different values"
        );
    }

    /// LLVM's named identifiers are `[-a-zA-Z$._][-a-zA-Z$._0-9]*`. A leading digit
    /// is the *unnamed* form, so a numeric hint is not a name that can be given —
    /// it is refused rather than silently colliding with the unnamed counter.
    #[test]
    fn a_numeric_hint_is_refused() {
        let (mut ctx, f, _) = two_functions();

        assert!(
            ctx.name_for_reg(Some("0"), f).is_err(),
            "`%0` is the unnamed form, not a name a caller may ask for"
        );

        assert_eq!(
            name(&mut ctx, None, f),
            "0",
            "and the refusal leaves the unnamed counter untouched"
        );
    }

    /// The rest of the grammar: anything outside the identifier charset would have
    /// to be quoted in the emitted IR, so it is refused here instead.
    #[test]
    fn a_hint_outside_the_identifier_grammar_is_refused() {
        let (mut ctx, f, _) = two_functions();

        for hint in ["my reg", "a+b", "", "a\"b", "café", "x\ny"] {
            assert!(
                ctx.name_for_reg(Some(hint), f).is_err(),
                "`{hint}` is not a legal unquoted LLVM identifier"
            );
        }
    }

    /// And the characters that *are* legal, including the ones that look unusual —
    /// `$`, `.`, `_` and `-` are all identifier characters, and a name may start
    /// with any of them.
    #[test]
    fn the_full_identifier_charset_is_accepted() {
        let (mut ctx, f, _) = two_functions();

        for hint in [
            "x", "_x", ".x", "$x", "-x", "x.y", "x_y", "x$y", "x-y", "x0", "A", "a1b2",
        ] {
            assert_eq!(
                name(&mut ctx, Some(hint), f),
                hint,
                "`{hint}` is a legal identifier and is unused, so it comes back as-is"
            );
        }
    }

    /// A `Context` owns the arenas an id indexes, so an id from another one is not
    /// silently written into the wrong block — `id_arena` catches it.
    #[test]
    #[should_panic(expected = "valid id are never constructed")]
    fn an_id_from_another_context_panics_rather_than_writing_elsewhere() {
        // A builder per context: a builder's duplicate-name set holds `StrId`s,
        // which only mean anything against the context they were interned in.
        let (mut ctx_a, mut builder_a) = fixture();
        let (mut ctx_b, mut builder_b) = fixture();

        let f = builder_a.add_function("f".to_string(), &mut ctx_a).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx_a).unwrap();
        let g = builder_b.add_function("g".to_string(), &mut ctx_b).unwrap();

        g.add_basic_block("entry".to_string(), &mut ctx_b).unwrap();

        // `entry` indexes `ctx_a`'s block arena, so reaching for it in `ctx_b` must
        // not land on whatever block sits at the same position there.
        let cursor = builder_a.cursor_at_block(entry);

        cursor.add_unconditional_br(entry, &mut ctx_b);
    }

    /// The string pool is shared across the whole context, so a name used by both
    /// a function and a block costs one entry.
    #[test]
    fn names_are_interned_once_across_the_context() {
        let (mut ctx, mut builder) = fixture();

        let f = builder
            .add_function("shared".to_string(), &mut ctx)
            .unwrap();

        f.add_basic_block("shared".to_string(), &mut ctx).unwrap();

        assert_eq!(
            ctx.str_interner.len(),
            1,
            "the function and the block share one pooled name"
        );

        let _ = Type::I1;
    }
}
