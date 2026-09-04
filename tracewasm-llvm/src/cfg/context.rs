//! Storage for everything a module's ids point into.

use crate::{
    cfg::{
        basic_block::{BasicBlock, BasicBlockId},
        builder::Builder,
        function::{FuncId, Function},
        module::{DataLayout, Module, Triple},
    },
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::ContextError,
    instruction::cursor::RegName,
    interner::{ConstInterner, StrId, StrInterner, TyId, TyInterner},
    value::{Type, TypeDisplay},
};
use id_arena::Arena;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;

/// Everything one module's ids point into: the block and function arenas, the three
/// interner pools, and the per-function register bookkeeping.
///
/// **An id only means anything against the context that issued it.** A [`TyId`] is a
/// position in *this* `ty_interner`, so resolving one against another context reads
/// whatever type happens to sit at that position — silently, since an id carries no
/// provenance. The same holds for [`StrId`] and the arena ids, though those at least
/// panic rather than acting on the wrong entry.
///
/// In practice: one context per module, threaded through every builder call. That is
/// why so much of this crate takes `&mut Context` rather than the individual pool it
/// happens to need.
pub struct Context {
    pub(crate) module: Module,
    pub(crate) blocks: Arena<BasicBlock>,
    pub(crate) funcs: Arena<Function>,
    pub(crate) str_interner: StrInterner,
    pub(crate) const_interner: ConstInterner,
    pub(crate) ty_interner: TyInterner,
    pub(crate) reg_name_assigner: FxHashMap<FuncId, FuncRegNameIndex>,
    pub(crate) register_def_instr_index: FxHashMap<FuncId, FxHashMap<StrId, RegisterDef>>,
}

impl Context {
    /// An empty context for the given target.
    ///
    /// One per module. Everything built against it is addressed by id, and an id only
    /// means anything here — see the type-level note above.
    pub fn new(triple: Triple, data_layout: DataLayout) -> Self {
        Context {
            module: Module::new(triple, data_layout),
            blocks: Arena::default(),
            funcs: Arena::default(),
            str_interner: StrInterner::default(),
            const_interner: ConstInterner::default(),
            ty_interner: TyInterner::default(),
            reg_name_assigner: FxHashMap::default(),
            register_def_instr_index: FxHashMap::default(),
        }
    }

    pub fn builder(self) -> Builder {
        Builder { ctx: self }
    }
}

/// Where a register was defined.
///
/// Both halves are needed. The index is a position in *that block's* instruction
/// list, so on its own it means nothing: read against another block it lands on an
/// unrelated instruction, or past the end. This is what
/// [`Value::try_inferring_pointee_ty`](crate::value::Value) follows to reach an
/// `alloca` in the entry block from a `store` several blocks later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegisterDef {
    /// The block holding the defining instruction.
    pub(crate) block: BasicBlockId,
    /// Its position in that block's instruction list.
    pub(crate) instr_index: usize,
}

impl Context {
    /// Issues a unique register name within `func_id`.
    ///
    /// With a hint, the name is the hint — suffixed if it is taken. Without one, the
    /// next unnamed index, which is how LLVM's `%0`, `%1`, … numbering is produced.
    ///
    /// The counter is per function and **parameters draw from it first**, so a body's
    /// first unnamed temporary continues where the parameter list stopped. A *named*
    /// parameter consumes nothing.
    ///
    /// # Errors
    ///
    /// [`ContextError::InvalidRegisterName`] if the hint is not a legal LLVM local.
    pub(crate) fn name_for_reg(
        &mut self,
        name: RegName,
        func_id: FuncId,
    ) -> Result<String, ContextError> {
        let assigner = self.reg_name_assigner.entry(func_id).or_default();
        let name = assigner.name_from_hint(name)?;

        Ok(name)
    }

    /// Resolves a block id. Panics only if the id came from another context.
    pub(crate) fn get_block(&self, id: BasicBlockId) -> &BasicBlock {
        self.blocks
            .get(id.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    /// Resolves a block id mutably. Panics only if the id came from another context.
    pub(crate) fn get_block_mut(&mut self, id: BasicBlockId) -> &mut BasicBlock {
        self.blocks
            .get_mut(id.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    /// Resolves a function id. Panics only if the id came from another context.
    pub(crate) fn get_func(&self, id: FuncId) -> &Function {
        self.funcs
            .get(id.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    /// Resolves a function id mutably. Panics only if the id came from another
    /// context.
    pub(crate) fn get_func_mut(&mut self, id: FuncId) -> &mut Function {
        self.funcs
            .get_mut(id.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    /// Every register `func` has defined, by name.
    ///
    /// The map is created by [`Builder::define_function`](crate::cfg::builder::Builder::define_function),
    /// so it exists — empty — from the moment the function does. That is what the
    /// `expect` here relies on; without it, a function whose first instruction reads
    /// a parameter pointer would panic before it could define anything.
    ///
    /// Parameters are *not* in the map: nothing in the function defines them.
    pub(crate) fn register_defs(&self, func: FuncId) -> &FxHashMap<StrId, RegisterDef> {
        self.register_def_instr_index
            .get(&func)
            .expect("this entry should exist for the func_id")
    }

    /// Interns `ptr` and returns its id.
    pub fn ptr_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::Ptr).into()
    }

    /// Borrows a type for printing. See [`TyId::display`](crate::interner::TyId).
    pub fn display(&self, id: TyId) -> TypeDisplay<'_> {
        id.display(self)
    }

    /// Interns `i1` and returns its id.
    pub fn i1_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::I1).into()
    }

    /// Interns `i8` and returns its id.
    pub fn i8_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::I8).into()
    }

    /// Interns `i16` and returns its id.
    pub fn i16_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::I16).into()
    }

    /// Interns `i32` and returns its id.
    pub fn i32_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::I32).into()
    }

    /// Interns `i64` and returns its id.
    pub fn i64_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::I64).into()
    }

    /// Interns `half` and returns its id.
    pub fn f16_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::Half).into()
    }

    /// Interns `float` and returns its id.
    pub fn f32_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::Float).into()
    }

    /// Interns `double` and returns its id.
    pub fn f64_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::Double).into()
    }

    /// Interns `void` and returns its id.
    pub fn void_ty(&mut self) -> TyId {
        self.ty_interner.intern(Type::Void).into()
    }
}

/// Hands out unique register names within one function.
///
/// Two schemes at once, matching LLVM: unnamed values get consecutive numbers from
/// `%0`, and a hinted name is used as given unless it is taken, in which case a
/// numeric suffix is appended. `issued_names` guards the case where a *hint* collides
/// with a suffix this assigner would generate — asking for `x` twice yields `x` and
/// `x1`, so a later request for `x1` must not produce a duplicate.
#[derive(Default)]
pub(crate) struct FuncRegNameIndex {
    unnamed_index: u32,
    named_index: FxHashMap<String, u32>,
    issued_names: FxHashSet<String>,
}

impl FuncRegNameIndex {
    /// The next `%N` for an unnamed value.
    fn next_unnamed_index(&mut self) -> u32 {
        let index = self.unnamed_index;

        self.unnamed_index += 1;

        index
    }

    /// How many times `name` has been asked for. `0` the first time, so the first
    /// request keeps the hint unsuffixed.
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

    /// Turns an optional hint into a name no other register in this function has.
    ///
    /// A hint must be a legal unquoted LLVM local — `[-a-zA-Z$._][-a-zA-Z$._0-9]*`.
    /// A leading digit is refused for a second reason beyond quoting: `%0` is the
    /// *unnamed* form, so a numeric hint would collide with the counter rather than
    /// merely need quotes.
    ///
    /// The loop retries suffixes until it finds one not already issued, which is what
    /// keeps a requested `x1` distinct from the `x1` generated for a second `x`.
    fn name_from_hint(&mut self, hint: RegName) -> Result<String, ContextError> {
        let RegName::Named(hint) = hint else {
            return Ok(self.next_unnamed_index().to_string());
        };

        let re = Regex::new(r"^[-a-zA-Z$._][-a-zA-Z$._0-9]*$").unwrap();

        if !re.is_match(&hint) {
            return Err(ContextError::InvalidRegisterName(hint.to_string()));
        }

        let final_name = loop {
            let index = self.next_named_index(&hint);

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
    use crate::{
        cfg::{
            builder::Builder,
            global::{DefinedFunc, GlobalId},
        },
        test_support::{add_fn, fixture},
        value::Type,
    };

    /// A context, a builder, and two functions to scope names against.
    fn two_functions() -> (Context, GlobalId<DefinedFunc>, GlobalId<DefinedFunc>) {
        let mut ctx = crate::test_support::ctx();
        let builder = Builder;

        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let g = add_fn("g", &builder, &mut ctx).unwrap();

        (ctx, f, g)
    }

    fn name(ctx: &mut Context, hint: RegName, func: GlobalId<DefinedFunc>) -> String {
        let spelled = match &hint {
            RegName::Named(n) => n.clone(),
            RegName::Unnamed => "<unnamed>".to_string(),
        };

        ctx.name_for_reg(hint, func.tag.raw())
            .unwrap_or_else(|e| panic!("hint `{spelled}` should be accepted: {e}"))
    }

    /// LLVM numbers unnamed temporaries from 0, in order, and the numbering is per
    /// function — `%0` is the first unnamed value in *that* function's body.
    #[test]
    fn unnamed_values_are_numbered_from_zero_per_function() {
        let (mut ctx, f, g) = two_functions();

        let in_f: Vec<String> = (0..3)
            .map(|_| name(&mut ctx, RegName::Unnamed, f))
            .collect();
        let in_g: Vec<String> = (0..2)
            .map(|_| name(&mut ctx, RegName::Unnamed, g))
            .collect();

        assert_eq!(in_f, ["0", "1", "2"]);
        assert_eq!(in_g, ["0", "1"], "a second function restarts at 0");
    }

    /// Local names are scoped to their function, so the same hint in two functions
    /// is not a collision and neither needs a suffix.
    #[test]
    fn a_hint_is_scoped_to_its_function() {
        let (mut ctx, f, g) = two_functions();

        assert_eq!(name(&mut ctx, "sum".into(), f), "sum");
        assert_eq!(
            name(&mut ctx, "sum".into(), g),
            "sum",
            "another function's `sum` is a different value"
        );
    }

    /// Within one function a name may be defined once, so a repeated hint has to
    /// come back changed.
    #[test]
    fn a_repeated_hint_is_made_unique() {
        let (mut ctx, f, _) = two_functions();

        let names: Vec<String> = (0..3).map(|_| name(&mut ctx, "x".into(), f)).collect();

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
            RegName::Named("x".to_string()),
            RegName::Named("x".to_string()),
            RegName::Named("x1".to_string()),
            RegName::Unnamed,
            RegName::Named("x".to_string()),
            RegName::Unnamed,
            RegName::Named("x2".to_string()),
        ];

        for hint in requests {
            let spelled = match &hint {
                RegName::Named(n) => n.clone(),
                RegName::Unnamed => "<unnamed>".to_string(),
            };

            let issued_name = name(&mut ctx, hint, f);

            assert!(
                issued.insert(issued_name.clone()),
                "`{issued_name}` was handed out twice (hint `{spelled}`); LLVM rejects \
                 two definitions of the same local"
            );
        }
    }

    /// A hint that already looks like a suffixed name must not collide with the
    /// suffix the assigner would generate.
    #[test]
    fn a_hint_matching_a_generated_suffix_does_not_collide() {
        let (mut ctx, f, _) = two_functions();

        let first = name(&mut ctx, "x".into(), f);
        let suffixed = name(&mut ctx, "x".into(), f);
        let asked_for = name(&mut ctx, "x1".into(), f);

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
            ctx.name_for_reg("0".into(), f.tag.raw()).is_err(),
            "`%0` is the unnamed form, not a name a caller may ask for"
        );

        assert_eq!(
            name(&mut ctx, RegName::Unnamed, f),
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
                ctx.name_for_reg(hint.into(), f.tag.raw()).is_err(),
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
                name(&mut ctx, hint.into(), f),
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
        let (mut ctx_a, builder_a) = fixture();
        let (mut ctx_b, builder_b) = fixture();

        let f = add_fn("f", &builder_a, &mut ctx_a).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx_a).unwrap();
        let g = add_fn("g", &builder_b, &mut ctx_b).unwrap();

        g.add_basic_block("entry".to_string(), &mut ctx_b).unwrap();

        // `entry` indexes `ctx_a`'s block arena, so reaching for it in `ctx_b` must
        // not land on whatever block sits at the same position there.
        let cursor = builder_a.cursor_at_block(entry);

        cursor.build_unconditional_br(entry, &mut ctx_b).unwrap();
    }

    /// The string pool is shared across the whole context, so a name used by both
    /// a function and a block costs one entry.
    #[test]
    fn names_are_interned_once_across_the_context() {
        let (mut ctx, builder) = fixture();

        let f = add_fn("shared", &builder, &mut ctx).unwrap();

        f.add_basic_block("shared".to_string(), &mut ctx).unwrap();

        assert_eq!(
            ctx.str_interner.len(),
            1,
            "the function and the block share one pooled name"
        );

        let _ = Type::I1;
    }
}
