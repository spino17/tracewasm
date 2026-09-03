//! The cursor: a position in a basic block, and the builders that write into it.

use crate::{
    cfg::{
        basic_block::BasicBlockId,
        context::{Context, RegisterDef},
        global::GlobalKind,
    },
    error::{
        AllocaError, CallError, FArithmeticError, FCmpError, GepError, IArithmeticError, ICmpError,
        InstructionError, PhiError, RetError, StoreError,
    },
    instruction::{
        AllocaOperands, CallOperands, ConditionalBrOperands, FArithmeticOp, FArithmeticOperands,
        FCmpOperands, FCond, FNegOperands, GetElementPtrOperands, IArithmeticOp,
        IArithmeticOperands, ICmpOperands, ICond, Instruction, InstructionKind, LoadOperands,
        PhiInstrHandler, PhiInstruction, RetOperands, StoreOperands, UnconditionalBrOperands,
    },
    interner::{StrId, TyId},
    value::{I1Value, Signedness, Value, ValueKind},
};
use rustc_hash::FxHashSet;

pub enum RegName {
    Named(String),
    Unnamed,
}

impl From<String> for RegName {
    fn from(value: String) -> Self {
        RegName::Named(value)
    }
}

impl From<&str> for RegName {
    fn from(value: &str) -> Self {
        RegName::Named(value.to_string())
    }
}

impl From<&String> for RegName {
    fn from(value: &String) -> Self {
        RegName::Named(value.to_string())
    }
}

pub enum OperandTy {
    Asserted(TyId),
    Inferred,
}

impl From<TyId> for OperandTy {
    fn from(value: TyId) -> Self {
        OperandTy::Asserted(value)
    }
}

/// Writes instructions into one basic block.
///
/// Obtained from [`Builder::cursor_at_block`](crate::cfg::builder::Builder::cursor_at_block).
/// Every builder appends to the end of the block, so instructions land in the order
/// they are added.
///
/// # Ending a block
///
/// The three terminator builders — [`build_unconditional_br`](Self::build_unconditional_br),
/// [`build_conditional_br`](Self::build_conditional_br) and [`build_ret`](Self::build_ret) —
/// take `self` **by value**. Once a block is ended, that cursor is gone, and the
/// compiler enforces it:
///
/// ```compile_fail
/// # use tracewasm_llvm::cfg::{builder::Builder, context::Context};
/// # let mut ctx = crate::test_support::ctx();
/// # let mut builder = Builder;
/// # let void_ty = ctx.void_ty();
/// # let f = builder.define_function("f".to_string(), &[], void_ty, &mut ctx).unwrap();
/// # let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
/// let cursor = builder.cursor_at_block(entry);
/// cursor.build_unconditional_br(entry, &mut ctx)?;
/// cursor.build_unconditional_br(entry, &mut ctx)?;  // cursor was moved
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// That cannot see a *second* cursor opened at the same block, so the block also
/// carries a flag: any builder on an ended block returns
/// [`InstructionError::BasicBlockAlreadyTerminated`].
///
/// # Naming
///
/// Builders that define a register take a `reg: Option<&str>` name hint. `None` gives
/// the value LLVM's next unnamed number; a hint is used as given, suffixed if it is
/// already taken.
pub struct Cursor {
    pub(crate) block: BasicBlockId,
}

impl Cursor {
    /// Builds a phi node, returning a handle for adding later branches and the register
    /// it defines.
    ///
    /// The phi's type comes from the **first** branch; every branch given here, and
    /// every one added later through the handle, is checked against it.
    ///
    /// # Errors
    ///
    /// - [`PhiError::PhiInstructionWithNoBranches`] — a phi with no incoming values
    ///   selects nothing, and there would be no type to give it.
    /// - [`PhiError::PhiInstructionCannotBeAddedToEntryBasicBlock`] — the entry block
    ///   has no predecessors.
    /// - [`PhiError::PhiInstructionAddError`] — phis come first in a block.
    /// - [`PhiError::BasicBlockAlreadyTerminated`] — the block already ended.
    /// - [`PhiError::PhiInstructionBranchTypeMismatch`] — a branch disagrees with the
    ///   phi's type.
    pub fn build_phi(
        &self,
        branches: &[(BasicBlockId, Value)],
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<(PhiInstrHandler, Value), PhiError> {
        if branches.is_empty() {
            return Err(PhiError::PhiInstructionWithNoBranches);
        }

        let ref_ty = branches[0].1.ty();
        let func_id = ctx.get_block(self.block).func_id;
        let reg_name = ctx.name_for_reg(reg, func_id)?;
        let val = Value::from_register(reg_name, ref_ty, ctx);

        let phi_id = self.block.add_phi(
            PhiInstruction {
                branches: vec![],
                blocks: FxHashSet::default(),
                ref_ty,
                value: val.clone(),
            },
            ctx,
        )?;

        for (branch, val) in branches {
            phi_id.add_branch((*branch, val.clone()), ctx)?;
        }

        Ok((phi_id, val))
    }

    /// Builds `br label %target`, ending the block.
    ///
    /// Consumes the cursor.
    ///
    /// # Errors
    ///
    /// [`InstructionError::BasicBlockAlreadyTerminated`] if a *different* cursor
    /// already ended this block.
    pub fn build_unconditional_br(
        self,
        label: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::UnconditionalBr(UnconditionalBrOperands { label }),
                value: None,
            },
            ctx,
        )?;

        self.block.set_locked(ctx);

        Ok(())
    }

    /// Builds `br i1 %c, label %t, label %f`, ending the block.
    ///
    /// Consumes the cursor. The condition is an [`I1Value`], so the `i1` requirement
    /// was already checked by [`Value::into_i1`](crate::value::Value::into_i1).
    ///
    /// # Errors
    ///
    /// [`InstructionError::BasicBlockAlreadyTerminated`] if a *different* cursor
    /// already ended this block.
    pub fn build_conditional_br(
        self,
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::ConditionalBr(ConditionalBrOperands {
                    cond,
                    true_label,
                    false_label,
                }),
                value: None,
            },
            ctx,
        )?;

        self.block.set_locked(ctx);

        Ok(())
    }

    /// Builds `ret <ty> %v` or `ret void`, ending the block.
    ///
    /// Consumes the cursor. The three well-formed shapes are:
    ///
    /// | `val` | `ty` | result |
    /// |---|---|---|
    /// | `Some(v)` | `Some(t)` | `v` folded into `t` |
    /// | `Some(v)` | `None` | type taken from `v` |
    /// | `None` | `Some(void)` | `ret void` |
    ///
    /// # Errors
    ///
    /// - [`RetError::DoesNotMatchFunctionResult`] — the return disagrees with the
    ///   enclosing function's declared result. Checked against the *function*, so
    ///   `ret i64 0` is refused inside an `i32` function even though the value and its
    ///   type agree with each other.
    /// - [`RetError::ValueGivenForVoid`] — `ret void` takes no value.
    /// - [`RetError::NonVoidTypeWithoutValue`] — a non-`void` return needs one.
    /// - [`RetError::TypeAndValueBothAbsent`] — nothing to return and nothing to infer
    ///   a type from.
    /// - [`RetError::ReturnedValueTypeMismatch`] — the value does not fold into `ty`.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] — the block already ended.
    pub fn build_ret(
        self,
        val: Option<&Value>,
        ty: OperandTy,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        let (val, ty) = match (val, ty) {
            (Some(val), OperandTy::Asserted(ty)) => {
                if ty.is_void(ctx) {
                    return Err(
                        RetError::ValueGivenForVoid(ctx.display(val.ty()).to_string()).into(),
                    );
                }

                let val_ty = ctx.display(val.ty()).to_string();

                let Some(casted_val) = val.try_cast(ty, Signedness::Signed, ctx) else {
                    return Err(RetError::ReturnedValueTypeMismatch(
                        val_ty,
                        ty.display(ctx).to_string(),
                    )
                    .into());
                };

                (Some(casted_val), ty)
            }
            (Some(val), OperandTy::Inferred) => {
                let ty = val.ty();

                (Some(val.clone()), ty)
            }
            (None, OperandTy::Asserted(ty)) => {
                if !ty.is_void(ctx) {
                    return Err(
                        RetError::NonVoidTypeWithoutValue(ty.display(ctx).to_string()).into(),
                    );
                }

                (None, ty)
            }
            (None, OperandTy::Inferred) => return Err(RetError::TypeAndValueBothAbsent.into()),
        };

        // LLVM checks the return against the *function's* result type, not just
        // against whatever the caller passed here: `define i32 @f() { ret void }` is
        // refused with "value doesn't match function result type".
        let func_id = ctx.get_block(self.block).func_id;
        let result_ty = ctx.get_func(func_id).result;

        if ty != result_ty {
            let func_name = ctx
                .str_interner
                .value(ctx.get_func(func_id).name.0)
                .to_string();

            return Err(RetError::DoesNotMatchFunctionResult(
                func_name,
                ctx.display(result_ty).to_string(),
                ctx.display(ty).to_string(),
            )
            .into());
        }

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Ret(RetOperands { ty, value: val }),
                value: None,
            },
            ctx,
        )?;

        self.block.set_locked(ctx);

        Ok(())
    }

    /// Builds `%x = alloca <ty>`, returning the pointer to the new slot.
    ///
    /// The result is a `ptr`, not `ty` — and the `alloca` records what it allocated,
    /// so a later `load` or `store` through that pointer can have its type inferred
    /// rather than restated.
    ///
    /// `count` allocates room for several elements: the value, and optionally a type
    /// to give it. `align` must be a power of two; `None` means the ABI default.
    ///
    /// # Errors
    ///
    /// - [`AllocaError::TypeNotAllocatable`] — `void` and function types have no size.
    /// - [`AllocaError::AllocaCountNotAnInteger`] — the count, or the type given for
    ///   it, is not an integer.
    /// - [`AllocaError::AllocaCountTypeMismatch`] — the count does not fold into the
    ///   type given for it.
    /// - [`InstructionError::AlignmentNotPowerOfTwo`] — including `0`; leaving
    ///   `align` off is how the default is asked for.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] — the block already ended.
    pub fn build_alloca(
        &self,
        ty: TyId,
        count: Option<(&Value, Option<TyId>)>,
        align: Option<u32>,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        if !ty.is_first_class(ctx) {
            return Err(AllocaError::TypeNotAllocatable(ty.display(ctx).to_string()).into());
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        let mut final_count: Option<Value> = None;

        if let Some((count_val, count_expected_ty)) = count {
            if !count_val.is_integer(ctx) {
                return Err(AllocaError::AllocaCountNotAnInteger(
                    ctx.display(count_val.ty()).to_string(),
                )
                .into());
            }

            let final_count_val = if let Some(count_expected_ty) = count_expected_ty {
                if !count_expected_ty.is_integer(ctx) {
                    return Err(AllocaError::AllocaCountNotAnInteger(
                        count_expected_ty.display(ctx).to_string(),
                    )
                    .into());
                }

                let count_ty = ctx.display(count_val.ty()).to_string();

                let Some(casted_val) =
                    count_val.try_cast(count_expected_ty, Signedness::Signed, ctx)
                else {
                    return Err(AllocaError::AllocaCountTypeMismatch(
                        count_ty,
                        count_expected_ty.display(ctx).to_string(),
                    )
                    .into());
                };

                casted_val
            } else {
                count_val.clone()
            };

            final_count = Some(final_count_val);
        }

        let result_ty = ctx.ptr_ty();

        add_instruction_to_block_and_get_value(
            InstructionKind::Alloca(AllocaOperands {
                ty,
                count: final_count,
                align,
            }),
            result_ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Builds `%x = load <ty>, ptr %p`, returning the loaded value.
    ///
    /// `ty` may be omitted when the pointer can be traced back to what it points at —
    /// an `alloca` or a `getelementptr`. A pointer that arrived as a function
    /// parameter cannot be, so there the type is required.
    ///
    /// # Errors
    ///
    /// - [`InstructionError::PointerOperandExpected`] — the operand is not a `ptr`.
    /// - [`InstructionError::TypeNotLoadable`] — `void` and function types have no
    ///   size. Aggregates do: `load {i32, i32}` assembles.
    /// - [`InstructionError::LoadedTypeDoesNotMatchPointee`] — the given type
    ///   disagrees with what the pointer was traced to. Stricter than LLVM, which
    ///   allows a load to reinterpret.
    /// - [`InstructionError::LoadedTypeUnknown`] — no type given and none inferable.
    /// - [`InstructionError::AlignmentNotPowerOfTwo`] — including `0`.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] — the block already ended.
    pub fn build_load(
        &self,
        ptr: &Value,
        ty: OperandTy,
        align: Option<u32>,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        let pointee_ty = ptr.try_inferring_pointee_ty(self.block, ctx);

        let final_ty = if let OperandTy::Asserted(ty) = ty {
            if !ty.is_first_class(ctx) {
                return Err(InstructionError::TypeNotLoadable(
                    ty.display(ctx).to_string(),
                ));
            }

            if let Some(pointee_ty) = pointee_ty
                && pointee_ty.ty != ty
            {
                return Err(InstructionError::LoadedTypeDoesNotMatchPointee(
                    ty.display(ctx).to_string(),
                    ctx.display(pointee_ty.ty).to_string(),
                ));
            }

            ty
        } else if let Some(pointee_ty) = pointee_ty {
            pointee_ty.ty
        } else {
            return Err(InstructionError::LoadedTypeUnknown);
        };

        add_instruction_to_block_and_get_value(
            InstructionKind::Load(LoadOperands {
                ty: final_ty,
                ptr: ptr.clone(),
                align,
            }),
            final_ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Builds `store <ty> %v, ptr %p`.
    ///
    /// Produces no value, so it defines no register. With `ty` given, the value is
    /// folded into it — a constant widens, a register must already match. Without,
    /// the value keeps its own type.
    ///
    /// # Errors
    ///
    /// - [`InstructionError::PointerOperandExpected`] — the destination is not a
    ///   `ptr`.
    /// - [`StoreError::StoredValueTypeMismatch`] — the value does not fold into `ty`.
    /// - [`StoreError::StoredValueDoesNotMatchPointee`] — the value disagrees with
    ///   what the pointer was traced to. Stricter than LLVM.
    /// - [`InstructionError::AlignmentNotPowerOfTwo`] — including `0`.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] — the block already ended.
    pub fn build_store(
        &self,
        ptr: &Value,
        value: &Value,
        ty: OperandTy,
        align: Option<u32>,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        let final_val = if let OperandTy::Asserted(ty) = ty {
            let value_ty = ctx.display(value.ty()).to_string();

            let Some(casted_value) = value.try_cast(ty, Signedness::Signed, ctx) else {
                return Err(StoreError::StoredValueTypeMismatch(
                    value_ty,
                    ty.display(ctx).to_string(),
                )
                .into());
            };

            casted_value
        } else {
            value.clone()
        };

        if let Some(pointee_ty) = ptr.try_inferring_pointee_ty(self.block, ctx)
            && pointee_ty.ty != final_val.ty()
        {
            return Err(StoreError::StoredValueDoesNotMatchPointee(
                ctx.display(final_val.ty()).to_string(),
                ctx.display(pointee_ty.ty).to_string(),
            )
            .into());
        }

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Store(StoreOperands {
                    value: final_val,
                    ptr: ptr.clone(),
                    align,
                }),
                value: None,
            },
            ctx,
        )?;

        Ok(())
    }

    /// Builds `%x = getelementptr <ty>, ptr %p, ...`, returning the computed pointer.
    ///
    /// Address arithmetic only — nothing is read from memory. The result is always a
    /// `ptr`, and it records what it points at, so a chain of `getelementptr`s can
    /// each infer their source type from the one before.
    ///
    /// # Indices
    ///
    /// The **first** index steps over `source_ty` as pointer arithmetic rather than
    /// into it: `getelementptr %T, ptr %p, i32 1` addresses the *next* `%T`. Only the
    /// remaining indices descend, one aggregate level each. So walking into field 1
    /// of a struct is `[0, 1]`, not `[1]`.
    ///
    /// `source_ty` may be omitted when the pointer can be traced back to what it
    /// points at; a pointer that arrived as a function parameter cannot be.
    ///
    /// # Errors
    ///
    /// - [`InstructionError::PointerOperandExpected`] — the base is not a `ptr`.
    /// - [`GepError::IndexNotAnInteger`] — every index scales an offset.
    /// - [`GepError::SourceTypeNotSized`] / [`GepError::SourceTypeUnknown`] /
    ///   [`GepError::SourceTypeDoesNotMatchPointee`] — about `source_ty` itself.
    /// - [`GepError::StructIndexNotAConstantI32`] — a struct index names a field, so
    ///   it must be known now and be an `i32` specifically.
    /// - [`GepError::StructIndexOutOfRange`] / [`GepError::ArrayIndexOutOfRange`] —
    ///   the array case is stricter than LLVM, which treats it as runtime UB.
    /// - [`GepError::TypeNotIndexable`] — indices remain but the walk reached a
    ///   scalar.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] — the block already ended.
    pub fn build_get_element_ptr(
        &self,
        ptr: &Value,
        source_ty: OperandTy,
        indices: &[Value],
        inbounds: Option<bool>,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let inbounds = inbounds.unwrap_or(false);

        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        for index in indices {
            if !index.is_integer(ctx) {
                return Err(
                    GepError::IndexNotAnInteger(ctx.display(index.ty()).to_string()).into(),
                );
            }
        }

        let pointee_ty = ptr.try_inferring_pointee_ty(self.block, ctx);

        let final_source_ty = if let OperandTy::Asserted(source_ty) = source_ty {
            if !source_ty.is_first_class(ctx) {
                return Err(
                    GepError::SourceTypeNotSized(source_ty.display(ctx).to_string()).into(),
                );
            }

            if let Some(pointee_ty) = pointee_ty
                && source_ty != pointee_ty.ty
            {
                return Err(GepError::SourceTypeDoesNotMatchPointee(
                    source_ty.display(ctx).to_string(),
                    ctx.display(pointee_ty.ty).to_string(),
                )
                .into());
            }

            source_ty
        } else if let Some(pointee_ty) = pointee_ty {
            pointee_ty.ty
        } else {
            return Err(GepError::SourceTypeUnknown.into());
        };

        // The first index steps over the source type as pointer arithmetic rather than
        // descending into it, so the walk starts at the second.
        if indices.len() > 1 {
            final_source_ty.walk_pointee_ty_in_gep(&indices[1..], ctx)?;
        }

        let result_ty = ctx.ptr_ty();

        add_instruction_to_block_and_get_value(
            InstructionKind::GetElementPtr(GetElementPtrOperands {
                // The resolved type, not the caller's `Option`: when it was inferred
                // from the pointer, that is the type the instruction has to be emitted
                // with, and it is the one the walk above validated.
                source_ty: final_source_ty,
                ptr: ptr.clone(),
                indices: indices.to_vec().into_boxed_slice(),
                inbounds,
            }),
            result_ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Builds `%x = call <ty> @f(...)`, returning the register it defines.
    ///
    /// For a callee that returns `void` use
    /// [`build_void_call`](Self::build_void_call) instead. The two are separate
    /// because a `void` call defines no register — `llvm-as` refuses
    /// `%r = call void @g()` with "instructions returning void cannot have a name" —
    /// so there is no [`Value`] for this one to hand back. Splitting them is what
    /// makes naming a `void` call unrepresentable rather than a runtime check on
    /// `reg`, and it is why this returns a `Value` rather than an `Option<Value>`.
    ///
    /// A `call` is not a terminator: the block stays open afterwards.
    ///
    /// # Checking against the callee
    ///
    /// The callee is looked up by name in the module's function table and everything
    /// is checked against its recorded signature — arity, each parameter type, and
    /// the return type. Each argument may carry an optional type to be folded into
    /// first, with the same rule as elsewhere: a constant converts, a register must
    /// already match.
    ///
    /// `return_ty` is optional; omitted, it is taken from the signature. Given, it
    /// has to agree with it.
    ///
    /// # The callee must already exist
    ///
    /// Only functions added through
    /// [`Builder::define_function`](crate::cfg::builder::Builder::define_function) are in
    /// the table, and they are added as the module is built. So:
    ///
    /// - **Self-recursion works** — a function's name is registered before its body
    ///   is built.
    /// - **A forward call does not.** Calling a function that will be added later
    ///   fails, even though LLVM makes every function in a module mutually visible.
    ///   Add all the functions first, then fill in their bodies.
    /// - **Host imports are reachable**, through
    ///   [`Builder::declare_function`](crate::cfg::builder::Builder::declare_function).
    ///   A declaration records its signature under the same name, so it resolves and
    ///   is checked exactly as a definition would be.
    ///
    /// # Errors
    ///
    /// - [`CallError::FunctionNotFound`] if no function of that name has been added.
    /// - [`CallError::VoidCalleeNeedsVoidCall`] if the callee returns `void` — use
    ///   [`build_void_call`](Self::build_void_call).
    /// - [`CallError::ParamCountMismatch`], [`CallError::ParamTypeMismatch`],
    ///   [`CallError::ParamCastFailed`] for the arguments.
    /// - [`CallError::ReturnTypeMismatch`] if `return_ty` disagrees with the
    ///   signature.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_call(
        &self,
        func_name: String,
        params: &[(&Value, Option<TyId>)],
        return_ty: OperandTy,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let params: Vec<(Value, Option<TyId>)> = params
            .iter()
            .copied()
            .map(|(x, y)| (x.clone(), y))
            .collect();

        let func_name_id: StrId = ctx.str_interner.intern(func_name).into();

        // The signature is read out by value before anything below borrows `ctx`
        // mutably: casting an argument interns into the type pool, which a live
        // borrow of the function table would forbid.
        let Some(global) = ctx.module.globals.get(&func_name_id) else {
            return Err(CallError::FunctionNotFound(
                ctx.str_interner.value(func_name_id.0).to_string(),
            )
            .into());
        };

        let GlobalKind::Func(func_sig) = &global.kind else {
            unreachable!("hitting this means globals tracking logic by their name is incorrect")
        };

        let name = ctx.str_interner.value(func_name_id.0).to_string();
        let expected_return_ty = func_sig.result;
        let expected_param_tys = func_sig.params.clone();

        let return_ty = if let OperandTy::Asserted(return_ty) = return_ty {
            if return_ty != expected_return_ty {
                return Err(CallError::ReturnTypeMismatch(
                    name,
                    ctx.display(expected_return_ty).to_string(),
                    ctx.display(return_ty).to_string(),
                )
                .into());
            }

            return_ty
        } else {
            expected_return_ty
        };

        // A `void` call defines no register, so it has no `Value` to hand back — and
        // `llvm-as` refuses `%r = call void @g()` outright. Splitting the two builders
        // is what makes that unrepresentable rather than a runtime check on `reg`.
        if return_ty.is_void(ctx) {
            return Err(CallError::VoidCalleeNeedsVoidCall(name).into());
        }

        let final_params =
            try_cast_param_and_check_with_func_signature(name, &params, &expected_param_tys, ctx)?;

        add_instruction_to_block_and_get_value(
            InstructionKind::Call(CallOperands {
                func_name: func_name_id,
                return_ty,
                params: final_params,
            }),
            return_ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Builds `call void @f(...)`, which defines no register.
    ///
    /// The counterpart to [`build_call`](Self::build_call), for a callee whose result
    /// is `void`. It takes no `reg` and returns nothing, because there is nothing to
    /// name: `llvm-as` refuses `%r = call void @g()` with "instructions returning void
    /// cannot have a name". Having the two as separate builders is what makes that
    /// mistake unrepresentable instead of a check that could be forgotten.
    ///
    /// Arguments are resolved and checked against the callee's signature exactly as in
    /// [`build_call`](Self::build_call), and the same rules apply for which callees are
    /// reachable.
    ///
    /// A `call` is not a terminator: the block stays open afterwards.
    ///
    /// # Errors
    ///
    /// - [`CallError::FunctionNotFound`] if no function of that name has been added.
    /// - [`CallError::NonVoidCalleeNeedsValueCall`] if the callee returns a value,
    ///   which this builder would silently drop — use
    ///   [`build_call`](Self::build_call).
    /// - [`CallError::ParamCountMismatch`], [`CallError::ParamTypeMismatch`],
    ///   [`CallError::ParamCastFailed`] for the arguments.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_void_call(
        &self,
        func_name: String,
        params: &[(&Value, Option<TyId>)],
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        let params: Vec<(Value, Option<TyId>)> = params
            .iter()
            .copied()
            .map(|(x, y)| (x.clone(), y))
            .collect();

        let func_name_id: StrId = ctx.str_interner.intern(func_name).into();

        // The signature is read out by value before anything below borrows `ctx`
        // mutably: casting an argument interns into the type pool, which a live
        // borrow of the function table would forbid.
        let Some(global) = ctx.module.globals.get(&func_name_id) else {
            return Err(CallError::FunctionNotFound(
                ctx.str_interner.value(func_name_id.0).to_string(),
            )
            .into());
        };

        let GlobalKind::Func(func_sig) = &global.kind else {
            unreachable!("hitting this means globals tracking logic by their name is incorrect")
        };

        let name = ctx.str_interner.value(func_name_id.0).to_string();
        let expected_return_ty = func_sig.result;
        let expected_param_tys = func_sig.params.clone();

        // The mirror of the check in `build_call`: a callee that returns something
        // would have its result silently dropped here, since this builder defines no
        // register and hands nothing back.
        if !expected_return_ty.is_void(ctx) {
            return Err(CallError::NonVoidCalleeNeedsValueCall(
                name,
                ctx.display(expected_return_ty).to_string(),
            )
            .into());
        }

        let final_params =
            try_cast_param_and_check_with_func_signature(name, &params, &expected_param_tys, ctx)?;

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Call(CallOperands {
                    func_name: func_name_id,
                    return_ty: ctx.void_ty(),
                    params: final_params,
                }),
                value: None,
            },
            ctx,
        )?;

        Ok(())
    }

    /// Compares two integers or pointers, defining an `i1`.
    ///
    /// Emits `icmp <cond> <ty> <a>, <b>`. The result is an [`I1Value`], so it can be
    /// handed straight to [`build_conditional_br`](Self::build_conditional_br) without
    /// a second check that it is an `i1`.
    ///
    /// # Operand types must agree
    ///
    /// LLVM requires it — `llvm-as` refuses `icmp slt i64 %a, %b` where `%b` is an
    /// `i32`. Where the predicate carries a signedness, a differing-width **constant**
    /// is widened to match, using `zext` for the `u*` predicates and `sext` for the
    /// `s*` ones. A differing-width *register* is always refused: widening one needs
    /// an instruction this builder will not insert on the caller's behalf.
    ///
    /// `eq` and `ne` carry no signedness and so widen nothing — they require operands
    /// that already share a type. That is not a limitation but the absence of an
    /// answer: against `i64 4294967295`, a `-1i32` operand is equal zero-extended and
    /// unequal sign-extended, and only the caller knows which was meant.
    ///
    /// # The `ty` argument means two things
    ///
    /// For the ordered predicates it **coerces**: both operands are cast to it. For
    /// `eq`/`ne` it **asserts**: it must equal the type the operands already have,
    /// since there is no signedness with which to coerce them.
    ///
    /// # Errors
    ///
    /// - [`ICmpError::OperandsNotCastable`] — no common type, or a constant that does
    ///   not fit one under the predicate's reading.
    /// - [`ICmpError::OperandTypesDiffer`] — `eq`/`ne` given operands of two types.
    /// - [`ICmpError::ProvidedTypeDoesNotMatchOperands`] — `eq`/`ne` given a `ty` its
    ///   operands do not have.
    /// - [`ICmpError::OperandTypeNotComparable`] — not an integer or a pointer.
    ///   Pointers are fine with every predicate; floats need `fcmp`.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_icmp(
        &self,
        cond: ICond,
        ty: OperandTy,
        a: &Value,
        b: &Value,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<I1Value, InstructionError> {
        let (a, b) = if let Some(signedness) = cond.signedness() {
            // Ids are `Copy`, so the operand types survive the move into the cast and
            // are rendered only on the failing path.
            let (given_a, given_b) = (a.ty(), b.ty());

            let Some((a, b)) = Value::try_cast_two(a, b, ty, signedness, ctx) else {
                return Err(ICmpError::OperandsNotCastable(
                    cond.to_string(),
                    ctx.display(given_a).to_string(),
                    ctx.display(given_b).to_string(),
                )
                .into());
            };

            (a, b)
        } else {
            if a.ty() != b.ty() {
                return Err(ICmpError::OperandTypesDiffer(
                    cond.to_string(),
                    ctx.display(a.ty()).to_string(),
                    ctx.display(b.ty()).to_string(),
                )
                .into());
            }

            if let OperandTy::Asserted(ty) = ty
                && ty != a.ty()
            {
                return Err(ICmpError::ProvidedTypeDoesNotMatchOperands(
                    cond.to_string(),
                    ctx.display(ty).to_string(),
                    ctx.display(a.ty()).to_string(),
                )
                .into());
            }

            (a.clone(), b.clone())
        };

        // Both operands share a type by here: the strict arm asserted it, and the cast
        // returns a pair that agrees. So checking one covers both.
        let ty = a.ty();

        if !ty.is_integer(ctx) && !ty.is_ptr(ctx) {
            return Err(ICmpError::OperandTypeNotComparable(ctx.display(ty).to_string()).into());
        }

        let val = add_instruction_to_block_and_get_value(
            InstructionKind::ICmp(ICmpOperands { cond, ty, a, b }),
            ctx.i1_ty(),
            self.block,
            reg,
            ctx,
        )?;

        let i1val = val
            .into_i1(ctx)
            .expect("the result type passed just above is i1");

        Ok(i1val)
    }

    /// Integer arithmetic, bitwise logic or a shift, defining a value of the operand
    /// type.
    ///
    /// Emits `<op> <ty> <a>, <b>`. Unlike a comparison, the result has the operands'
    /// type rather than `i1`.
    ///
    /// # Operand types must agree
    ///
    /// The six operations LLVM spells with a signedness — `sdiv`/`udiv`, `srem`/`urem`
    /// and `ashr`/`lshr` — widen a narrower **constant** to match, zero- or
    /// sign-extending as that operation says.
    ///
    /// The other seven refuse to widen at all. `add`, `sub`, `mul`, `shl`, `and`, `or`
    /// and `xor` have a single opcode each because the result bits are the same under
    /// either reading — which means nothing says how to *widen* an operand, and the
    /// two choices give different answers. So they require operands that already share
    /// a type, exactly as `icmp eq` does.
    ///
    /// A differing-width *register* is always refused: widening one needs an
    /// instruction this builder will not insert on the caller's behalf.
    ///
    /// # Errors
    ///
    /// - [`IArithmeticError::OperandsNotCastable`] — a signed operation whose operands
    ///   have no common type.
    /// - [`IArithmeticError::OperandTypesDiffer`] — a signedness-free operation given
    ///   two types.
    /// - [`IArithmeticError::ProvidedTypeDoesNotMatchOperands`] — a signedness-free
    ///   operation given a `ty` its operands do not have.
    /// - [`IArithmeticError::OperandTypeNotInteger`] — floats need the `f`-prefixed
    ///   instructions.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_iarithmetic(
        &self,
        op: IArithmeticOp,
        ty: OperandTy,
        a: &Value,
        b: &Value,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let (a, b) = if let Some(signedness) = op.signedness() {
            // Ids are `Copy`, so the operand types survive the move into the cast and
            // are rendered only on the failing path.
            let (given_a, given_b) = (a.ty(), b.ty());

            let Some((a, b)) = Value::try_cast_two(a, b, ty, signedness, ctx) else {
                return Err(IArithmeticError::OperandsNotCastable(
                    op.to_string(),
                    ctx.display(given_a).to_string(),
                    ctx.display(given_b).to_string(),
                )
                .into());
            };

            (a, b)
        } else {
            if a.ty() != b.ty() {
                return Err(IArithmeticError::OperandTypesDiffer(
                    op.to_string(),
                    ctx.display(a.ty()).to_string(),
                    ctx.display(b.ty()).to_string(),
                )
                .into());
            }

            if let OperandTy::Asserted(ty) = ty
                && ty != a.ty()
            {
                return Err(IArithmeticError::ProvidedTypeDoesNotMatchOperands(
                    op.to_string(),
                    ctx.display(ty).to_string(),
                    ctx.display(a.ty()).to_string(),
                )
                .into());
            }

            (a.clone(), b.clone())
        };

        // Both operands share a type by here, so checking one covers both.
        let ty = a.ty();

        if !ty.is_integer(ctx) {
            return Err(IArithmeticError::OperandTypeNotInteger(
                op.to_string(),
                ctx.display(ty).to_string(),
            )
            .into());
        }

        add_instruction_to_block_and_get_value(
            InstructionKind::IArithmetic(IArithmeticOperands { op, ty, a, b }),
            ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Compares two floating-point values, defining an `i1`.
    ///
    /// Emits `fcmp <cond> <ty> <a>, <b>`. The result is an [`I1Value`], so it can be
    /// handed straight to [`build_conditional_br`](Self::build_conditional_br).
    ///
    /// # Operand types must agree
    ///
    /// As with [`build_icmp`](Self::build_icmp), LLVM requires it. A differing-width
    /// **constant** is widened to match; a differing-width *register* is refused,
    /// since widening one needs an `fpext` this builder will not insert.
    ///
    /// No signedness is involved — a float carries its sign in its format, so `fpext`
    /// is exact and there is nothing for a caller to choose. That is why the cast runs
    /// under [`Signedness::NotApplicable`], which also refuses
    /// every integer cast and so keeps an integer from reaching a float comparison by
    /// way of a silent widening.
    ///
    /// # Errors
    ///
    /// - [`FCmpError::OperandsNotCastable`] — no common type. Integer operands land
    ///   here, since nothing bridges the two families without a real `sitofp`.
    /// - [`FCmpError::OperandTypeNotFloat`] — the common type is not a float.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_fcmp(
        &self,
        cond: FCond,
        ty: OperandTy,
        a: &Value,
        b: &Value,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<I1Value, InstructionError> {
        // Ids are `Copy`, so the operand types survive the move into the cast and are
        // rendered only on the failing path.
        let (given_a, given_b) = (a.ty(), b.ty());

        let Some((a, b)) = Value::try_cast_two(a, b, ty, Signedness::NotApplicable, ctx) else {
            return Err(FCmpError::OperandsNotCastable(
                cond.to_string(),
                ctx.display(given_a).to_string(),
                ctx.display(given_b).to_string(),
            )
            .into());
        };

        // Both operands share a type by here, so checking one covers both.
        let ty = a.ty();

        if !ty.is_float(ctx) {
            return Err(FCmpError::OperandTypeNotFloat(ctx.display(ty).to_string()).into());
        }

        let val = add_instruction_to_block_and_get_value(
            InstructionKind::FCmp(FCmpOperands { cond, ty, a, b }),
            ctx.i1_ty(),
            self.block,
            reg,
            ctx,
        )?;

        let i1val = val
            .into_i1(ctx)
            .expect("the result type passed just above is i1");

        Ok(i1val)
    }

    /// Floating-point arithmetic, defining a value of the operand type.
    ///
    /// Emits `<op> <ty> <a>, <b>`. A narrower float **constant** widens to match, and
    /// needs no signedness to do it: `fpext` is exact, so there is nothing for a
    /// caller to choose. The cast therefore runs under
    /// [`Signedness::NotApplicable`], which also refuses every
    /// integer cast and so keeps an integer from reaching a float instruction by way
    /// of a silent widening.
    ///
    /// For negation see [`build_fneg`](Self::build_fneg) — `fneg` is unary.
    ///
    /// # Errors
    ///
    /// - [`FArithmeticError::OperandsNotCastable`] — no common type. Integer operands
    ///   land here.
    /// - [`FArithmeticError::OperandTypeNotFloat`] — the common type is not a float.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_farithmetic(
        &self,
        op: FArithmeticOp,
        ty: OperandTy,
        a: &Value,
        b: &Value,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let (given_a, given_b) = (a.ty(), b.ty());

        let Some((a, b)) = Value::try_cast_two(a, b, ty, Signedness::NotApplicable, ctx) else {
            return Err(FArithmeticError::OperandsNotCastable(
                op.to_string(),
                ctx.display(given_a).to_string(),
                ctx.display(given_b).to_string(),
            )
            .into());
        };

        let ty = a.ty();

        if !ty.is_float(ctx) {
            return Err(FArithmeticError::OperandTypeNotFloat(
                op.to_string(),
                ctx.display(ty).to_string(),
            )
            .into());
        }

        add_instruction_to_block_and_get_value(
            InstructionKind::FArithmetic(FArithmeticOperands { op, ty, a, b }),
            ty,
            self.block,
            reg,
            ctx,
        )
    }

    /// Negates a floating-point value, defining one of the same type.
    ///
    /// Emits `fneg <ty> <value>`. Separate from
    /// [`build_farithmetic`](Self::build_farithmetic) because `fneg` takes a **single**
    /// operand — `llvm-as` refuses `fneg double %a, %b`. There is no integer
    /// counterpart; negating one is `sub 0, %x`.
    ///
    /// # Errors
    ///
    /// - [`FArithmeticError::OperandTypeNotFloat`] if the operand is not a float.
    /// - [`InstructionError::BasicBlockAlreadyTerminated`] if the block is closed.
    pub fn build_fneg(
        &self,
        value: Value,
        reg: RegName,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let ty = value.ty();

        if !ty.is_float(ctx) {
            return Err(FArithmeticError::OperandTypeNotFloat(
                "fneg".to_string(),
                ctx.display(ty).to_string(),
            )
            .into());
        }

        add_instruction_to_block_and_get_value(
            InstructionKind::FNeg(FNegOperands { ty, value }),
            ty,
            self.block,
            reg,
            ctx,
        )
    }
}

/// Resolves a call's arguments against the callee's parameter types.
///
/// Shared by both call builders, which differ only in what they do with the *result* —
/// the argument rules are identical either way.
///
/// Arity is checked first, so a mismatched count is reported as such rather than as a
/// type error on whichever argument happens to line up wrongly. Each argument may
/// carry an optional type to be folded into first, with the same rule as elsewhere: a
/// constant converts, a register must already match.
fn try_cast_param_and_check_with_func_signature(
    name: String,
    params: &[(Value, Option<TyId>)],
    expected_param_tys: &[TyId],
    ctx: &mut Context,
) -> Result<Vec<Value>, CallError> {
    if params.len() != expected_param_tys.len() {
        return Err(CallError::ParamCountMismatch {
            name,
            expected: expected_param_tys.len(),
            given: params.len(),
        });
    }

    let mut final_params: Vec<Value> = Vec::with_capacity(params.len());

    for (index, ((param_val, param_ty), expected_param_ty)) in
        params.iter().zip(expected_param_tys).enumerate()
    {
        let final_val = if let Some(param_ty) = param_ty {
            let given = ctx.display(param_val.ty()).to_string();

            let Some(casted_val) = param_val.try_cast(*param_ty, Signedness::Signed, ctx) else {
                return Err(CallError::ParamCastFailed(
                    name,
                    index,
                    given,
                    ctx.display(*param_ty).to_string(),
                ));
            };

            casted_val
        } else {
            param_val.clone()
        };

        if final_val.ty() != *expected_param_ty {
            return Err(CallError::ParamTypeMismatch(
                name,
                index,
                ctx.display(*expected_param_ty).to_string(),
                ctx.display(final_val.ty()).to_string(),
            ));
        }

        final_params.push(final_val);
    }

    Ok(final_params)
}

/// Appends an instruction that defines a register, and records where it was defined.
///
/// Shared by every value-producing builder. Recording the definition — block *and*
/// index — is what later lets [`Value::try_inferring_pointee_ty`](crate::value::Value)
/// walk back from a pointer to the instruction that produced it. The index alone
/// would be meaningless, since each block numbers its instructions from zero.
fn add_instruction_to_block_and_get_value(
    kind: InstructionKind,
    result_ty: TyId,
    block: BasicBlockId,
    reg: RegName,
    ctx: &mut Context,
) -> Result<Value, InstructionError> {
    let func_id = ctx.get_block(block).func_id;
    let reg_name = ctx.name_for_reg(reg, func_id)?;
    let val = Value::from_register(reg_name, result_ty, ctx);

    let ValueKind::Reg(reg) = val.kind() else {
        unreachable!("value is made out of register name just above")
    };

    let reg_name_id = reg.name;

    let instr_index = block.add_instruction(
        Instruction {
            kind,
            value: Some(val.clone()),
        },
        ctx,
    )?;

    let register_def_index = ctx.register_def_instr_index.entry(func_id).or_default();

    register_def_index.insert(reg_name_id, RegisterDef { block, instr_index });

    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg::{builder::Builder, context::Context},
        test_support::{add_fn, fixture, value},
        value::{ConstExpr, ConstValue, NullPtr, Type},
    };

    /// Interns `ty` and hands back its id, for the shapes that have no `Context`
    /// shorthand.
    fn intern(ty: Type, ctx: &mut Context) -> TyId {
        ctx.ty_interner.intern(ty).into()
    }

    /// How a type spells itself against `ctx`'s pool — which is the form an error
    /// carries, since `BuildError` holds the rendering rather than the type.
    fn rendered(ty: TyId, ctx: &Context) -> String {
        ty.display(ctx).to_string()
    }

    /// Instructions land in the block the cursor names, in order.
    #[test]
    fn instructions_append_in_order_to_the_open_block() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();
        let cursor = builder.cursor_at_block(entry);

        // A non-terminator first, so the block is still open for the branch that
        // follows it — a branch consumes the cursor and closes the block.
        cursor
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .unwrap();
        cursor.build_unconditional_br(body, &mut ctx).unwrap();

        let instrs = &ctx.blocks.get(entry.raw()).unwrap().instructions;

        assert_eq!(instrs.len(), 2);

        assert!(
            matches!(instrs[0].kind, InstructionKind::Alloca(_)),
            "the first instruction is the one added first"
        );

        assert!(
            matches!(
                instrs[1].kind,
                InstructionKind::UnconditionalBr(UnconditionalBrOperands{ label }) if label == body
            ),
            "and the branch is second"
        );

        assert!(
            instrs[1].value.is_none(),
            "a branch produces no value, so it defines no register"
        );

        assert!(
            ctx.blocks.get(body.raw()).unwrap().instructions.is_empty(),
            "the other block is untouched"
        );
    }

    /// A phi selects between incoming values, so with none there is nothing to
    /// select — and no first branch to take the phi's type from either.
    ///
    /// This is the one phi path that does not reach `Value::ty`, so it is the only
    /// one runnable until that lands.
    #[test]
    fn a_phi_needs_at_least_one_branch() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(body);

        let err = cursor
            .build_phi(&[], "result".into(), &mut ctx)
            .expect_err("a phi with no branches has no value and no type");

        assert!(matches!(err, PhiError::PhiInstructionWithNoBranches));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// The entry block has no predecessors, so a phi there has nothing to choose
    /// between.
    #[test]
    fn a_phi_cannot_go_in_the_entry_block() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(entry);

        let err = cursor
            .build_phi(&[(entry, v)], "result".into(), &mut ctx)
            .expect_err("no predecessors to choose between");

        assert!(matches!(
            err,
            PhiError::PhiInstructionCannotBeAddedToEntryBasicBlock
        ));
    }

    /// In a later block a phi is fine, and each one gets its own index — tagged
    /// with the block, since an index alone would mean different phis in different
    /// blocks.
    #[test]
    fn phis_in_a_later_block_are_indexed_within_that_block() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let tail = f.add_basic_block("tail".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let cursor = builder.cursor_at_block(body);

        let (first, _) = cursor
            .build_phi(&[(entry, v1)], "a".into(), &mut ctx)
            .unwrap();
        let (second, _) = cursor
            .build_phi(&[(entry, v2)], "b".into(), &mut ctx)
            .unwrap();

        assert_eq!((first.index, first.block), (0, body));
        assert_eq!((second.index, second.block), (1, body));

        // A different block starts its own numbering, which is why the id has to
        // carry the block to be unambiguous.
        let cursor = builder.cursor_at_block(tail);
        let (elsewhere, _) = cursor
            .build_phi(&[(entry, v3)], "c".into(), &mut ctx)
            .unwrap();

        assert_eq!((elsewhere.index, elsewhere.block), (0, tail));
        assert_ne!(elsewhere.block, first.block);
    }

    /// The phi's own type comes from its first branch, and the value it hands back
    /// carries that type — that is what makes the result usable downstream.
    #[test]
    fn a_phi_yields_a_value_of_its_branches_type() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(body);

        let (_, result) = cursor
            .build_phi(&[(entry, v)], "merged".into(), &mut ctx)
            .unwrap();

        assert_eq!(
            ctx.ty_interner.value(result.ty().raw()),
            &Type::I32,
            "an i32 branch makes an i32 phi"
        );
    }

    /// A phi produces one value, so every incoming value has to have its type.
    #[test]
    fn phi_branches_must_all_share_the_phis_type() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let an_i32 = value(1, &mut ctx);
        let an_i64 = Value::from_const(1i64, None, &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(body);

        let err = cursor
            .build_phi(
                &[(entry, an_i32), (other, an_i64)],
                "merged".into(),
                &mut ctx,
            )
            .expect_err("the second branch is an i64");

        assert!(
            matches!(
                &err,
                PhiError::PhiInstructionBranchTypeMismatch(phi, branch)
                    if phi == "i32" && branch == "i64"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// The other side of the check above: two branches whose types were built
    /// *separately* must still be accepted, because structurally equal types are one
    /// pool entry and so one id.
    ///
    /// This is what a non-number array length would break. If `Type::Array` held a
    /// `Value`, the two `[4 x i32]`s below would carry different constant ids, intern
    /// as two entries, and this phi would be refused as a type mismatch — valid IR
    /// rejected, with an error naming the same type twice.
    #[test]
    fn a_phi_accepts_branches_whose_equal_types_were_built_separately() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        let i32_ty = intern(Type::I32, &mut ctx);

        let array = |ctx: &mut Context| {
            intern(
                Type::Array {
                    size: 4,
                    element_ty: i32_ty,
                },
                ctx,
            )
        };

        // Built one at a time, as two unrelated parts of a module would build them.
        let first_ty = array(&mut ctx);
        let second_ty = array(&mut ctx);

        assert_eq!(first_ty, second_ty, "`[4 x i32]` is one type, hence one id");

        let a = Value::from_register("a".to_string(), first_ty, &mut ctx);
        let b = Value::from_register("b".to_string(), second_ty, &mut ctx);

        let cursor = builder.cursor_at_block(body);

        let (_, merged) = cursor
            .build_phi(&[(entry, a), (other, b)], "merged".into(), &mut ctx)
            .expect("both branches are `[4 x i32]`");

        assert_eq!(
            ctx.display(merged.ty()).to_string(),
            "[4 x i32]",
            "and the phi carries that type"
        );
    }

    /// Phis have to precede every other instruction in their block, so once one
    /// has been emitted the window has closed.
    #[test]
    fn a_phi_cannot_follow_an_instruction() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let i32_ty = ctx.i32_ty();
        let cursor = builder.cursor_at_block(body);

        // A plain instruction, not a terminator: this is about phis coming *first*,
        // not about the block being closed, which is a separate rule.
        cursor
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .unwrap();

        let err = cursor
            .build_phi(&[(entry, v)], "result".into(), &mut ctx)
            .expect_err("the block already has an instruction");

        assert!(matches!(err, PhiError::PhiInstructionAddError));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// A block stays open until a terminator: plain instructions keep appending, and
    /// nothing locks it.
    #[test]
    fn a_block_stays_open_until_a_terminator() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        let cursor = builder.cursor_at_block(entry);

        for _ in 0..3 {
            cursor
                .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
                .expect("a block that has not branched still accepts instructions");
        }

        assert!(
            !ctx.blocks.get(entry.raw()).unwrap().is_locked,
            "only a terminator locks a block"
        );

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().instructions.len(), 3);
    }

    /// Both terminators lock, and locking is per *block*: closing one leaves its
    /// siblings open.
    #[test]
    fn a_terminator_locks_its_own_block_only() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let exit = f.add_basic_block("exit".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_locked);
        assert!(
            !ctx.blocks.get(body.raw()).unwrap().is_locked,
            "a branch closes the block it was written into, not the one it names"
        );

        let cond = Value::from_const(true, None, &mut ctx)
            .unwrap()
            .into_i1(&ctx)
            .unwrap();

        builder
            .cursor_at_block(body)
            .build_conditional_br(cond, exit, exit, &mut ctx)
            .unwrap();

        assert!(
            ctx.blocks.get(body.raw()).unwrap().is_locked,
            "a conditional branch locks too"
        );

        assert!(!ctx.blocks.get(exit.raw()).unwrap().is_locked);
    }

    /// Consuming the cursor stops *that* cursor being reused, but a fresh one can
    /// still be opened at the same block — which is what `is_locked` is for, and the
    /// only thing that catches this: the consumed-`self` rule cannot see a second
    /// cursor.
    #[test]
    fn a_fresh_cursor_on_a_locked_block_is_refused() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        // A second cursor at the same block: the consumed-`self` rule cannot see this.
        let reopened = builder.cursor_at_block(entry);

        let err = reopened
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .expect_err("`entry` already ends in a branch");

        assert!(
            matches!(&err, InstructionError::BasicBlockAlreadyTerminated(name) if name == "entry"),
            "the error must name the block it refused, got: {err}"
        );

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "the refused instruction must not have been added"
        );
    }

    /// The same on the phi path. The lock check runs *before* the
    /// `instructions.is_empty()` one, so a terminated block reports being terminated
    /// rather than the vaguer `PhiInstructionAddError` — which is the more useful of
    /// the two, since both are true of a block that has branched.
    #[test]
    fn a_phi_on_a_locked_block_reports_the_block_is_terminated() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);

        builder
            .cursor_at_block(body)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        let reopened = builder.cursor_at_block(body);

        let err = reopened
            .build_phi(&[(entry, v)], "result".into(), &mut ctx)
            .expect_err("`body` already ends in a branch");

        assert!(
            matches!(&err, PhiError::BasicBlockAlreadyTerminated(name) if name == "body"),
            "the error must name the block it refused, got: {err}"
        );

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// Every builder that writes an instruction goes through the same lock, not just
    /// the one that happened to be tested first.
    #[test]
    fn a_locked_block_refuses_every_kind_of_instruction() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        let i32_ty = ctx.i32_ty();
        let ptr = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        let terminated = |r: Result<(), InstructionError>, what: &str| {
            assert!(
                matches!(
                    &r,
                    Err(InstructionError::BasicBlockAlreadyTerminated(name)) if name == "entry"
                ),
                "`{what}` must be refused by the lock, got: {r:?}"
            );
        };

        terminated(
            builder
                .cursor_at_block(entry)
                .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
                .map(|_| ()),
            "alloca",
        );

        terminated(
            builder
                .cursor_at_block(entry)
                .build_load(&ptr, i32_ty.into(), None, RegName::Unnamed, &mut ctx)
                .map(|_| ()),
            "load",
        );

        terminated(
            builder.cursor_at_block(entry).build_store(
                &ptr,
                &seven,
                OperandTy::Inferred,
                None,
                &mut ctx,
            ),
            "store",
        );

        terminated(
            builder
                .cursor_at_block(entry)
                .build_get_element_ptr(
                    &ptr,
                    i32_ty.into(),
                    &[zero],
                    None,
                    RegName::Unnamed,
                    &mut ctx,
                )
                .map(|_| ()),
            "getelementptr",
        );

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "nothing was appended past the terminator"
        );
    }

    /// A second terminator is refused too, so a block ends in exactly one branch —
    /// which is what LLVM requires of a well-formed block.
    #[test]
    fn a_locked_block_refuses_a_second_terminator() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        let err = builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .expect_err("a block ends once");

        assert!(
            matches!(&err, InstructionError::BasicBlockAlreadyTerminated(name) if name == "entry"),
            "got: {err}"
        );

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().instructions.len(), 1);
    }

    /// The lock survives reopening: asking for a fresh cursor does not clear it, so a
    /// block cannot be reopened by going back to the builder.
    #[test]
    fn reopening_a_block_does_not_clear_its_lock() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        builder
            .cursor_at_block(entry)
            .build_unconditional_br(body, &mut ctx)
            .unwrap();

        for attempt in 0..3 {
            let reopened = builder.cursor_at_block(entry);

            assert!(
                reopened
                    .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
                    .is_err(),
                "attempt {attempt} must still be refused"
            );

            assert!(ctx.blocks.get(entry.raw()).unwrap().is_locked);
        }
    }

    /// A phi still goes in before the block is closed — locking is about the
    /// terminator, and an open block accepts phis and instructions as before.
    #[test]
    fn locking_does_not_disturb_a_well_formed_block() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let i32_ty = ctx.i32_ty();

        let cursor = builder.cursor_at_block(body);

        cursor
            .build_phi(&[(entry, v)], "m".into(), &mut ctx)
            .expect("a phi opens the block");

        cursor
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .expect("instructions follow the phi");

        cursor
            .build_unconditional_br(body, &mut ctx)
            .expect("and the terminator closes it");

        let block = ctx.blocks.get(body.raw()).unwrap();

        assert_eq!(block.phis.len(), 1);
        assert_eq!(block.instructions.len(), 2);
        assert!(block.is_locked, "closed only at the end");
    }

    /// A `ret` has to match the *function's* result type, not merely the type the
    /// caller passed alongside the value. `llvm-as` refuses every mismatch with
    /// "value doesn't match function result type".
    #[test]
    fn a_ret_must_match_the_function_result() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function("f".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        // `define i32 @f() { ret void }`
        let err = builder
            .cursor_at_block(entry)
            .build_ret(None, void_ty.into(), &mut ctx)
            .expect_err("an i32 function does not return void");

        assert!(
            matches!(
                &err,
                InstructionError::Ret(RetError::DoesNotMatchFunctionResult(name, want, got))
                    if name == "f" && want == "i32" && got == "void"
            ),
            "the error must name the function and both types, got: {err}"
        );

        // `ret i64 0` from the same function is refused for the same reason, even
        // though the value and its declared type agree with each other.
        let wide = Value::from_const(1i64, None, &mut ctx).unwrap();

        assert!(
            matches!(
                builder
                    .cursor_at_block(entry)
                    .build_ret(Some(&wide), i64_ty.into(), &mut ctx),
                Err(InstructionError::Ret(RetError::DoesNotMatchFunctionResult(
                    ..
                )))
            ),
            "a self-consistent value of the wrong width is still the wrong result"
        );

        assert!(
            !ctx.blocks.get(entry.raw()).unwrap().is_locked,
            "a refused ret must not close the block"
        );
    }

    /// The shapes a `ret` is built from: an explicit type with a matching value, the
    /// type inferred from the value, and `void` with no value at all.
    #[test]
    fn a_ret_accepts_its_three_well_formed_shapes() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let returns_i32 = builder
            .define_function("a".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let with_ty = returns_i32
            .add_basic_block("with_ty".to_string(), &mut ctx)
            .unwrap();

        let inferred = returns_i32
            .add_basic_block("inferred".to_string(), &mut ctx)
            .unwrap();

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();
        let eight = Value::from_const(8i32, None, &mut ctx).unwrap();

        builder
            .cursor_at_block(with_ty)
            .build_ret(Some(&seven), i32_ty.into(), &mut ctx)
            .expect("the type and the value agree");

        builder
            .cursor_at_block(inferred)
            .build_ret(Some(&eight), OperandTy::Inferred, &mut ctx)
            .expect("with no type given it comes from the value");

        let returns_void = builder
            .define_function("b".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let empty = returns_void
            .add_basic_block("entry".to_string(), &mut ctx)
            .unwrap();

        builder
            .cursor_at_block(empty)
            .build_ret(None, void_ty.into(), &mut ctx)
            .expect("`ret void` takes no value");

        for block in [with_ty, inferred, empty] {
            assert!(
                ctx.blocks.get(block.raw()).unwrap().is_locked,
                "a ret terminates its block"
            );
        }
    }

    /// The malformed combinations, each with its own error rather than one catch-all.
    #[test]
    fn a_ret_refuses_its_malformed_shapes() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let a = f.add_basic_block("a".to_string(), &mut ctx).unwrap();
        let b = f.add_basic_block("b".to_string(), &mut ctx).unwrap();
        let c = f.add_basic_block("c".to_string(), &mut ctx).unwrap();

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        // `ret void` with a value.
        let err = builder
            .cursor_at_block(a)
            .build_ret(Some(&seven), void_ty.into(), &mut ctx)
            .expect_err("`void` takes no value");

        assert!(
            matches!(
                &err,
                InstructionError::Ret(RetError::ValueGivenForVoid(t)) if t == "i32"
            ),
            "got: {err}"
        );

        // A non-`void` type with no value.
        let err = builder
            .cursor_at_block(b)
            .build_ret(None, i32_ty.into(), &mut ctx)
            .expect_err("`i32` needs a value");

        assert!(
            matches!(
                &err,
                InstructionError::Ret(RetError::NonVoidTypeWithoutValue(t)) if t == "i32"
            ),
            "got: {err}"
        );

        // Neither: there is nothing to infer a return from.
        let err = builder
            .cursor_at_block(c)
            .build_ret(None, OperandTy::Inferred, &mut ctx)
            .expect_err("neither a type nor a value");

        assert!(
            matches!(
                &err,
                InstructionError::Ret(RetError::TypeAndValueBothAbsent)
            ),
            "got: {err}"
        );
    }

    /// A `ret` locks its block like the branches do, so nothing follows it.
    #[test]
    fn a_ret_locks_its_block() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .build_ret(None, void_ty.into(), &mut ctx)
            .unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_locked);

        let err = builder
            .cursor_at_block(entry)
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .expect_err("`entry` already returned");

        assert!(
            matches!(&err, InstructionError::BasicBlockAlreadyTerminated(name) if name == "entry"),
            "got: {err}"
        );
    }

    /// A pointer that arrives as a **parameter** has no defining instruction, so
    /// inference declines rather than failing — and the caller's explicit type stands.
    ///
    /// This is the first instruction in the function on purpose: the definition map is
    /// keyed by function, and before this was fixed the entry did not exist until
    /// something had defined a register, so reaching here panicked.
    #[test]
    fn a_parameter_pointer_declines_inference_rather_than_panicking() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let ptr_ty = ctx.ptr_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[(ptr_ty, Some("base".to_string()))],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);
        let base = f
            .nth_param(0, &ctx)
            .expect("the function has one parameter")
            .clone();

        assert!(
            base.try_inferring_pointee_ty(entry, &mut ctx).is_none(),
            "nothing in this function defined the parameter, so there is nothing to walk back to"
        );

        cursor
            .build_load(&base, i32_ty.into(), None, "v".into(), &mut ctx)
            .expect("the explicit type stands when inference declines");
    }

    /// Because inference declines, one base pointer can be read and written at several
    /// different types — the check only fires when a pointee is actually known.
    #[test]
    fn a_parameter_pointer_accepts_several_types_through_one_base() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[(ptr_ty, Some("base".to_string()))],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);
        let base = f
            .nth_param(0, &ctx)
            .expect("the function has one parameter")
            .clone();

        for (ty, reg) in [(i32_ty, "a"), (i64_ty, "b"), (f64_ty, "c")] {
            let loaded = cursor
                .build_load(&base, ty.into(), None, reg.into(), &mut ctx)
                .unwrap_or_else(|e| panic!("loading through a parameter must work: {e}"));

            assert_eq!(loaded.ty(), ty, "the load has the type it was given");
        }

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        cursor
            .build_store(&base, &seven, OperandTy::Inferred, None, &mut ctx)
            .expect("storing through a parameter works for the same reason");

        // And a `getelementptr` needs its source type given, since there is none to
        // infer — but it is accepted once supplied.
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();

        assert!(
            matches!(
                cursor.build_get_element_ptr(
                    &base,
                    OperandTy::Inferred,
                    std::slice::from_ref(&zero),
                    None,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::Gep(GepError::SourceTypeUnknown))
            ),
            "with nothing to infer from, a source type is required"
        );

        cursor
            .build_get_element_ptr(&base, i32_ty.into(), &[zero], None, "g".into(), &mut ctx)
            .expect("and supplying it is enough");
    }

    /// The definition map has an entry for a function as soon as it exists, not only
    /// once something has defined a register in it. That is the invariant
    /// `Context::register_defs` asserts, and it has to hold from the start.
    #[test]
    fn a_function_has_a_definition_map_before_any_instruction() {
        let (mut ctx, builder) = fixture();

        let void_ty = ctx.void_ty();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        assert!(
            ctx.register_defs(f.tag.raw()).is_empty(),
            "the map exists and is empty before anything is built"
        );

        // A second function gets its own, so the entry is per function rather than
        // created once by whoever built first.
        let g = builder
            .define_function("g".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        builder
            .cursor_at_block(entry)
            .build_alloca(i32_ty, None, None, "x".into(), &mut ctx)
            .unwrap();

        assert_eq!(
            ctx.register_defs(f.tag.raw()).len(),
            1,
            "`f` recorded its alloca"
        );

        assert!(
            ctx.register_defs(g.tag.raw()).is_empty(),
            "`g` is untouched"
        );
    }

    /// A `void` callee defines no register, and a non-`void` one does. `llvm-as`
    /// refuses `%r = call void @g()` with "instructions returning void cannot have a
    /// name", so asking for one is an error rather than something quietly dropped.
    #[test]
    fn a_call_defines_a_register_only_when_the_callee_returns_one() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let returns_i32 = builder
            .define_function("a".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        builder
            .define_function("b".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = returns_i32
            .add_basic_block("entry".to_string(), &mut ctx)
            .unwrap();

        let cursor = builder.cursor_at_block(entry);

        let value = cursor
            .build_call(
                "a".to_string(),
                &[],
                OperandTy::Inferred,
                "r".into(),
                &mut ctx,
            )
            .unwrap();

        assert_eq!(value.ty(), i32_ty, "typed by the callee's result");

        // A `void` callee has no value to define, so it belongs to the other builder.
        // That is no longer a check on `reg` — `build_void_call` takes no register
        // name at all, so naming a void call is not expressible.
        cursor
            .build_void_call("b".to_string(), &[], &mut ctx)
            .expect("a void callee is what this builder is for");
    }

    /// The two call builders each refuse the other's callee, so a result is never
    /// silently dropped and a `void` call never asks for a register it cannot have.
    ///
    /// `llvm-as` refuses `%r = call void @g()` with "instructions returning void
    /// cannot have a name"; splitting the builders is what turns that into a shape
    /// the caller cannot write.
    #[test]
    fn each_call_builder_refuses_the_other_kind_of_callee() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let host = builder
            .define_function("host".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        builder
            .define_function("nothing".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = host.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let err = cursor
            .build_call(
                "nothing".to_string(),
                &[],
                OperandTy::Inferred,
                "x".into(),
                &mut ctx,
            )
            .expect_err("a void callee has no value to return");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::VoidCalleeNeedsVoidCall(n)) if n == "nothing"
            ),
            "the error must name the callee and point at the other builder: {err}"
        );

        let err = cursor
            .build_void_call("host".to_string(), &[], &mut ctx)
            .expect_err("an i32 callee's result would be dropped");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::NonVoidCalleeNeedsValueCall(n, ty))
                    if n == "host" && ty == "i32"
            ),
            "the error must name the callee and its result type: {err}"
        );
    }

    /// The callee has to already be in the module. Self-recursion works, because a
    /// function's name is registered before its body is built; a **forward** call
    /// does not, even though LLVM makes every function in a module mutually visible.
    #[test]
    fn a_call_resolves_only_against_functions_already_added() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();

        let f = builder
            .define_function("f".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        assert!(
            cursor
                .build_call(
                    "f".to_string(),
                    &[],
                    OperandTy::Inferred,
                    "rec".into(),
                    &mut ctx
                )
                .is_ok(),
            "a function may call itself"
        );

        let err = cursor
            .build_call(
                "later".to_string(),
                &[],
                OperandTy::Inferred,
                "fwd".into(),
                &mut ctx,
            )
            .expect_err("`later` has not been added");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::FunctionNotFound(n)) if n == "later"
            ),
            "got: {err}"
        );
    }

    /// Arity, argument types and the return type are all checked against the callee's
    /// recorded signature.
    #[test]
    fn a_call_is_checked_against_the_callee_signature() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();

        builder
            .define_function("takes_i32".to_string(), &[(i32_ty, None)], i32_ty, &mut ctx)
            .unwrap();

        let f = builder
            .define_function("f".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        // Arity, checked before the argument types so a miscount reads as one.
        let err = cursor
            .build_call(
                "takes_i32".to_string(),
                &[],
                OperandTy::Inferred,
                "a".into(),
                &mut ctx,
            )
            .expect_err("it takes one argument");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::ParamCountMismatch { name, expected, given })
                    if name == "takes_i32" && *expected == 1 && *given == 0
            ),
            "got: {err}"
        );

        // A register of the wrong width is refused rather than widened.
        let wide = Value::from_register("w".to_string(), i64_ty, &mut ctx);

        let err = cursor
            .build_call(
                "takes_i32".to_string(),
                &[(&wide, None)],
                OperandTy::Inferred,
                "b".into(),
                &mut ctx,
            )
            .expect_err("an i64 is not an i32");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::ParamTypeMismatch(name, index, want, got))
                    if name == "takes_i32" && *index == 0 && want == "i32" && got == "i64"
            ),
            "got: {err}"
        );

        // A declared return type that disagrees with the callee's.
        let err = cursor
            .build_call(
                "takes_i32".to_string(),
                &[(&seven, None)],
                f64_ty.into(),
                "c".into(),
                &mut ctx,
            )
            .expect_err("it returns i32");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::ReturnTypeMismatch(name, want, got))
                    if name == "takes_i32" && want == "i32" && got == "double"
            ),
            "got: {err}"
        );

        // And the well-formed call still goes through.
        assert!(
            cursor
                .build_call(
                    "takes_i32".to_string(),
                    &[(&seven, None)],
                    i32_ty.into(),
                    "d".into(),
                    &mut ctx
                )
                .is_ok(),
            "a matching call is accepted"
        );
    }

    /// An argument may carry a type to be folded into first — a constant converts, a
    /// register has to match already.
    #[test]
    fn a_call_argument_may_be_cast_before_it_is_checked() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();

        builder
            .define_function("takes_i64".to_string(), &[(i64_ty, None)], i64_ty, &mut ctx)
            .unwrap();

        let f = builder
            .define_function("f".to_string(), &[], i64_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        // An `i32` constant widens into the `i64` the callee wants.
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        assert!(
            cursor
                .build_call(
                    "takes_i64".to_string(),
                    &[(&seven, Some(i64_ty))],
                    OperandTy::Inferred,
                    "a".into(),
                    &mut ctx
                )
                .is_ok(),
            "a constant folds into the declared type"
        );

        // But a cast that cannot happen at all is reported as the cast failing,
        // rather than as a type mismatch after the fact.
        let err = cursor
            .build_call(
                "takes_i64".to_string(),
                &[(&seven, Some(f64_ty))],
                OperandTy::Inferred,
                "b".into(),
                &mut ctx,
            )
            .expect_err("an integer does not become a double without an instruction");

        assert!(
            matches!(
                &err,
                InstructionError::Call(CallError::ParamCastFailed(name, index, from, to))
                    if name == "takes_i64" && *index == 0 && from == "i32" && to == "double"
            ),
            "got: {err}"
        );

        // A register is only checked, never converted.
        let narrow = Value::from_register("n".to_string(), i32_ty, &mut ctx);

        assert!(
            matches!(
                cursor.build_call(
                    "takes_i64".to_string(),
                    &[(&narrow, Some(i64_ty))],
                    OperandTy::Inferred,
                    "c".into(),
                    &mut ctx
                ),
                Err(InstructionError::Call(CallError::ParamCastFailed(..)))
            ),
            "widening a register needs a real instruction"
        );
    }

    /// A `call` is not a terminator, so the block stays open after one.
    #[test]
    fn a_call_does_not_end_its_block() {
        let (mut ctx, builder) = fixture();

        let void_ty = ctx.void_ty();
        let i32_ty = ctx.i32_ty();

        builder
            .define_function("g".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        cursor
            .build_void_call("g".to_string(), &[], &mut ctx)
            .unwrap();

        assert!(
            !ctx.blocks.get(entry.raw()).unwrap().is_locked,
            "a call is not a terminator"
        );

        cursor
            .build_alloca(i32_ty, None, None, RegName::Unnamed, &mut ctx)
            .expect("the block is still open");
    }

    /// A phi names one value per *predecessor*, so the same predecessor twice is a
    /// bug in the caller — and it is a different bug from an entry-block phi.
    #[test]
    fn a_phi_takes_each_predecessor_once() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let cursor = builder.cursor_at_block(body);

        let (phi, _) = cursor
            .build_phi(&[(entry, v1), (other, v2)], "merged".into(), &mut ctx)
            .unwrap();

        let err = phi
            .add_branch((entry, v3), &mut ctx)
            .expect_err("`entry` is already a predecessor of this phi");

        assert!(
            matches!(err, PhiError::BasicBlockBranchAlreadyInPhiInstruction),
            "a repeated predecessor is not an entry-block error, got: {err}"
        );

        // One incoming value per predecessor, and the two branches it was built
        // with are the two it has — not four.
        let stored = &ctx.blocks.get(body.raw()).unwrap().phis[phi.index];

        assert_eq!(stored.branches.len(), 2);
        assert_eq!(stored.blocks.len(), 2);
    }

    /// A block with a pointer-typed value in hand, which every load test needs.
    fn block_with_ptr(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let ptr = Value::from_const(NullPtr, None, ctx).unwrap();

        (builder.cursor_at_block(entry), ptr)
    }

    /// A `{ i32, double }` slot on the stack, plus the cursor to build against — the
    /// shape every struct-indexing test below needs.
    fn block_with_struct_slot(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value, TyId) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();

        let struct_ty = intern(
            Type::Struct {
                fields: Box::new([i32_ty, f64_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .build_alloca(struct_ty, None, None, "s".into(), ctx)
            .expect("a struct is allocatable");

        (cursor, slot, struct_ty)
    }

    /// A `{ i32, [4 x double] }` slot, for the tests that chain a `getelementptr`
    /// through a nested aggregate. Returns the cursor, the slot, and the two types.
    fn block_with_nested_slot(
        ctx: &mut Context,
        builder: &mut Builder,
    ) -> (Cursor, Value, TyId, TyId) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();

        let array_ty = intern(
            Type::Array {
                size: 4,
                element_ty: f64_ty,
            },
            ctx,
        );

        let struct_ty = intern(
            Type::Struct {
                fields: Box::new([i32_ty, array_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .build_alloca(struct_ty, None, None, "s".into(), ctx)
            .expect("a struct is allocatable");

        (cursor, slot, array_ty, f64_ty)
    }

    /// A `getelementptr` produces a pointer, so the *next* one has to be able to trace
    /// its pointee back through it — which is what makes a chain like
    /// `gep` → `gep` → `store` work without the caller restating the type each time.
    ///
    /// Before `GetElementPtr` was handled in the inference, this panicked rather than
    /// resolving: only `alloca` was matched.
    #[test]
    fn a_gep_through_a_gep_infers_its_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, array_ty, _) = block_with_nested_slot(&mut ctx, &mut builder);

        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        // `%f = gep { i32, [4 x double] }, ptr %s, i32 0, i32 1` — the array field.
        let field = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &[zero.clone(), one],
                None,
                "f".into(),
                &mut ctx,
            )
            .expect("the alloca says what it points to");

        // `%e = gep [4 x double], ptr %f, i32 0, i32 2` — with no source type given,
        // so it has to come from the gep above.
        let elem = cursor
            .build_get_element_ptr(
                &field,
                OperandTy::Inferred,
                &[zero, two],
                None,
                "e".into(),
                &mut ctx,
            )
            .expect("the first gep says what it points to");

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::GetElementPtr(second) = &block.instructions[2].kind else {
            panic!("expected a gep")
        };

        assert_eq!(
            second.source_ty, array_ty,
            "the second gep's source type is the first one's result pointee"
        );

        // And the chain lands on a `double`, which is what a store through it must be.
        let a_double = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let an_i32 = Value::from_const(1i32, None, &mut ctx).unwrap();

        assert!(
            cursor
                .build_store(&elem, &a_double, OperandTy::Inferred, None, &mut ctx)
                .is_ok(),
            "the element is a double"
        );

        let err = cursor
            .build_store(&elem, &an_i32, OperandTy::Inferred, None, &mut ctx)
            .expect_err("an i32 is not what this points to");

        assert!(
            matches!(
                &err,
                InstructionError::Store(StoreError::StoredValueDoesNotMatchPointee(value, pointee))
                    if value == "i32" && pointee == "double"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// The same inference from a `getelementptr` *constant expression* rather than an
    /// instruction — a constant expression records its operands just as the
    /// instruction does, so the pointee is equally recoverable.
    #[test]
    fn a_gep_constant_expression_reports_its_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _, array_ty, f64_ty) = block_with_nested_slot(&mut ctx, &mut builder);

        let ptr_ty = ctx.ptr_ty();
        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        // `getelementptr ([4 x double], ptr null, i32 0, i32 2)` — points at a double.
        let const_gep = Value::new(
            ptr_ty,
            ValueKind::ConstExpr(ConstExpr::GetElementPtr(Box::new(GetElementPtrOperands {
                source_ty: array_ty,
                ptr: null,
                indices: Box::new([zero, two]),
                inbounds: false,
            }))),
        );

        let pointee = const_gep
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("a gep constant expression says what it points to");

        assert_eq!(pointee.ty, f64_ty, "index 2 of `[4 x double]` is a double");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "double",
            "and it renders as one"
        );
    }

    /// A `getelementptr` with one index or none does not descend: the first index
    /// steps over the source type as pointer arithmetic, so the pointee is unchanged.
    #[test]
    fn a_gep_with_a_single_index_still_points_at_its_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _, _) = block_with_nested_slot(&mut ctx, &mut builder);

        let one = Value::from_const(1i32, None, &mut ctx).unwrap();

        // `%n = gep { i32, [4 x double] }, ptr %s, i32 1` — the *next* struct along.
        let next = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &[one],
                None,
                "n".into(),
                &mut ctx,
            )
            .expect("a single index is pointer arithmetic over the source type");

        let pointee = next
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("still traceable");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "{ i32, [4 x double] }",
            "one index steps over the type rather than into it"
        );
    }

    /// The types of the nested fixture below, so an assertion can name a level by
    /// what it means rather than by an id.
    struct DeepTys {
        outer: TyId,
        inner_array: TyId,
        inner: TyId,
        i64_array: TyId,
        i64: TyId,
        f64: TyId,
        ptr: TyId,
        i32: TyId,
    }

    /// A slot holding
    ///
    /// ```text
    /// %Inner = { i8, [3 x i64] }
    /// %Outer = { i32, [2 x %Inner], ptr, double }
    /// ```
    ///
    /// which is deep enough that an off-by-one in the walk lands on a real type
    /// rather than falling off the end — the failure mode a two-level fixture cannot
    /// catch. The index sequences the tests below use were checked against `llvm-as`:
    /// `0,1,1,1,2` assembles and one index further is refused, which is what pins the
    /// last level to a scalar.
    fn block_with_deep_slot(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value, DeepTys) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i8_ty = ctx.i8_ty();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();

        let i64_array = intern(
            Type::Array {
                size: 3,
                element_ty: i64_ty,
            },
            ctx,
        );

        let inner = intern(
            Type::Struct {
                fields: Box::new([i8_ty, i64_array]),
                packed: false,
            },
            ctx,
        );

        let inner_array = intern(
            Type::Array {
                size: 2,
                element_ty: inner,
            },
            ctx,
        );

        let outer = intern(
            Type::Struct {
                fields: Box::new([i32_ty, inner_array, ptr_ty, f64_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .build_alloca(outer, None, None, "s".into(), ctx)
            .expect("a struct is allocatable");

        (
            cursor,
            slot,
            DeepTys {
                outer,
                inner_array,
                inner,
                i64_array,
                i64: i64_ty,
                f64: f64_ty,
                ptr: ptr_ty,
                i32: i32_ty,
            },
        )
    }

    /// A constant `i32` index, which is what a struct index has to be.
    fn idx(n: i32, ctx: &mut Context) -> Value {
        Value::from_const(n, None, ctx).expect("an i32 constant")
    }

    /// Every level of the nested type, asserted by both id and spelling.
    ///
    /// One index per level after the first, so an off-by-one anywhere in the walk
    /// moves *every* row below it — which is why each row names the exact type rather
    /// than just checking the walk succeeded.
    #[test]
    fn a_gep_walks_every_level_of_a_deeply_nested_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        // The slot itself, before any walking, is the whole `%Outer`.
        assert_eq!(
            slot.try_inferring_pointee_ty(cursor.block, &mut ctx)
                .expect("an alloca says what it points to")
                .ty,
            tys.outer
        );

        // (indices after the leading 0, expected id, expected spelling)
        let cases: Vec<(Vec<i32>, TyId, &str)> = vec![
            (vec![0], tys.i32, "i32"),
            (vec![1], tys.inner_array, "[2 x { i8, [3 x i64] }]"),
            (vec![1, 1], tys.inner, "{ i8, [3 x i64] }"),
            (vec![1, 1, 1], tys.i64_array, "[3 x i64]"),
            (vec![1, 1, 1, 2], tys.i64, "i64"),
            (vec![2], tys.ptr, "ptr"),
            (vec![3], tys.f64, "double"),
        ];

        for (tail, expected_id, expected) in cases {
            let mut indices = vec![idx(0, &mut ctx)];

            for n in &tail {
                indices.push(idx(*n, &mut ctx));
            }

            let elem = cursor
                .build_get_element_ptr(
                    &slot.clone(),
                    OperandTy::Inferred,
                    &indices,
                    None,
                    RegName::Unnamed,
                    &mut ctx,
                )
                .unwrap_or_else(|e| panic!("gep 0,{tail:?} should walk: {e}"));

            let pointee = elem
                .try_inferring_pointee_ty(cursor.block, &mut ctx)
                .unwrap_or_else(|| panic!("gep 0,{tail:?} should report a pointee"));

            assert_eq!(
                ctx.display(pointee.ty).to_string(),
                expected,
                "gep 0,{tail:?} lands on the wrong type"
            );

            assert_eq!(
                pointee.ty, expected_id,
                "gep 0,{tail:?} renders as `{expected}` but is not that pool entry"
            );

            assert_eq!(
                ctx.display(elem.ty()).to_string(),
                "ptr",
                "a gep always yields a pointer"
            );
        }
    }

    /// One index past the last level names nothing. `llvm-as` agrees: appending an
    /// index to either sequence below is refused with "invalid getelementptr indices".
    #[test]
    fn a_gep_cannot_index_past_the_last_level() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_deep_slot(&mut ctx, &mut builder);

        // `0,1,1,1,2` ends on the `i64`; a sixth index has nothing to descend into.
        let too_deep = vec![
            idx(0, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(2, &mut ctx),
            idx(0, &mut ctx),
        ];

        let err = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &too_deep,
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("an i64 has no elements");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::TypeNotIndexable(t)) if t == "i64"),
            "the error must name the scalar it stopped at, got: {err}"
        );

        // And the same one level up, on the `double` field.
        let past_double = vec![idx(0, &mut ctx), idx(3, &mut ctx), idx(0, &mut ctx)];

        assert!(
            matches!(
                cursor.build_get_element_ptr(&slot, OperandTy::Inferred, &past_double, None, RegName::Unnamed, &mut ctx),
                Err(InstructionError::Gep(GepError::TypeNotIndexable(t))) if t == "double"
            ),
            "a double has no elements either"
        );
    }

    /// A `load` in the middle of a chain: the scalar a deep `getelementptr` reaches is
    /// loadable at exactly the type the walk landed on.
    #[test]
    fn a_load_through_a_deep_gep_yields_the_walked_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        let deep = &[
            idx(0, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(2, &mut ctx),
        ];

        let elem = cursor
            .build_get_element_ptr(&slot, OperandTy::Inferred, deep, None, "e".into(), &mut ctx)
            .expect("the walk reaches the i64");

        let loaded = cursor
            .build_load(&elem, tys.i64.into(), None, "v".into(), &mut ctx)
            .expect("an i64 is loadable");

        assert_eq!(ctx.display(loaded.ty()).to_string(), "i64");
        assert_eq!(loaded.ty(), tys.i64);

        // Storing the loaded value straight back is the round trip, and the pointee
        // check has to accept it.
        assert!(
            cursor
                .build_store(&elem, &loaded, OperandTy::Inferred, None, &mut ctx)
                .is_ok(),
            "what was loaded from a slot must store back into it"
        );
    }

    /// A `load` of a *pointer* breaks the chain, which is the honest answer: the
    /// pointee of a pointer read out of memory is not knowable from the instruction
    /// that produced it, so the caller has to say what it is.
    #[test]
    fn a_gep_through_a_loaded_pointer_needs_an_explicit_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        // `%p = gep %Outer, ptr %s, i32 0, i32 2` — the `ptr` field.
        let ptr_field = vec![idx(0, &mut ctx), idx(2, &mut ctx)];

        let ptr_field = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &ptr_field,
                None,
                "p".into(),
                &mut ctx,
            )
            .expect("field 2 is the pointer");

        // `%q = load ptr, ptr %p` — a pointer whose pointee nothing records.
        let loaded_ptr = cursor
            .build_load(&ptr_field, tys.ptr.into(), None, "q".into(), &mut ctx)
            .expect("a ptr is loadable");

        assert!(
            loaded_ptr
                .try_inferring_pointee_ty(cursor.block, &mut ctx)
                .is_none(),
            "a `load` says nothing about what its result points to"
        );

        let indices = vec![idx(0, &mut ctx), idx(1, &mut ctx)];

        let err = cursor
            .build_get_element_ptr(
                &loaded_ptr.clone(),
                OperandTy::Inferred,
                &indices,
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("nothing can be inferred through a load");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::SourceTypeUnknown)),
            "expected an unknown-source-type error, got: {err}"
        );

        // Given the type explicitly, the same walk goes through and lands where the
        // nested fixture says it should.
        let indices = vec![idx(0, &mut ctx), idx(1, &mut ctx)];

        let elem = cursor
            .build_get_element_ptr(
                &loaded_ptr,
                tys.inner.into(),
                &indices,
                None,
                "r".into(),
                &mut ctx,
            )
            .expect("with the source type given there is nothing to infer");

        let pointee = elem
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("the gep itself records its source type");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "[3 x i64]",
            "field 1 of `{{ i8, [3 x i64] }}` is the array"
        );

        assert_eq!(pointee.ty, tys.i64_array);
    }

    /// Memory is reached through a pointer, so `getelementptr` refuses anything else
    /// for the same reason `load` does.
    #[test]
    fn a_gep_needs_a_pointer_operand() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let not_a_ptr = value(7, &mut ctx);
        let zero = value(0, &mut ctx);

        let err = cursor
            .build_get_element_ptr(
                &not_a_ptr,
                i32_ty.into(),
                &[zero],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("an i32 is not an address");

        assert!(
            matches!(&err, InstructionError::PointerOperandExpected(t) if t == "i32"),
            "the error must name the offending type, got: {err}"
        );
    }

    /// Every index scales an offset, so every one of them has to be an integer.
    #[test]
    fn a_gep_index_must_be_an_integer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let a_float = Value::from_const(1.0f32, None, &mut ctx).unwrap();

        let err = cursor
            .build_get_element_ptr(
                &ptr,
                i32_ty.into(),
                &[a_float],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("a float is not an index");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::IndexNotAnInteger(t)) if t == "float"),
            "the error must name the offending type, got: {err}"
        );
    }

    /// With no source type given and a pointer nothing can be traced back to — a
    /// `null` constant — there is no type to emit the instruction with.
    #[test]
    fn a_gep_without_a_source_type_needs_an_inferable_pointer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);

        let err = cursor
            .build_get_element_ptr(
                &ptr,
                OperandTy::Inferred,
                &[zero],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("null says nothing about its pointee");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::SourceTypeUnknown)),
            "expected an unknown-source-type error, got: {err}"
        );
    }

    /// When the pointer *can* be traced back, a source type that disagrees with it is
    /// refused rather than silently preferred.
    #[test]
    fn a_gep_source_type_must_match_the_inferred_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let zero = value(0, &mut ctx);

        let err = cursor
            .build_get_element_ptr(
                &slot,
                i32_ty.into(),
                &[zero],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("the slot holds a struct, not an i32");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::SourceTypeDoesNotMatchPointee(given, inferred))
                    if given == "i32" && inferred == "{ i32, double }"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// Omitting the source type is allowed when the pointer says what it points to —
    /// and the *inferred* type is what the instruction is emitted with, not `None`.
    #[test]
    fn a_gep_emits_the_inferred_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, struct_ty) = block_with_struct_slot(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);
        let field = Value::from_const(1i32, None, &mut ctx).unwrap();

        let elem = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &[zero, field],
                None,
                "f".into(),
                &mut ctx,
            )
            .expect("the pointee is inferable from the alloca");

        assert_eq!(
            ctx.display(elem.ty()).to_string(),
            "ptr",
            "a gep yields a pointer"
        );

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::GetElementPtr(GetElementPtrOperands {
            source_ty,
            ptr: _,
            indices: _,
            inbounds: _,
        }) = &block.instructions[1].kind
        else {
            panic!("expected a gep")
        };

        assert_eq!(
            *source_ty, struct_ty,
            "the resolved type is emitted, not the caller's `None`"
        );
    }

    /// LLVM requires a struct index to be a constant `i32` specifically — `llvm-as`
    /// refuses an `i64` one with "invalid getelementptr indices" — even though array
    /// indices may be any width, because the index names a field rather than scaling
    /// an offset.
    #[test]
    fn a_struct_index_must_be_a_constant_i32() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let zero = value(0, &mut ctx);
        let wide = Value::from_const(1i64, None, &mut ctx).unwrap();

        let err = cursor
            .build_get_element_ptr(
                &slot.clone(),
                OperandTy::Inferred,
                &[zero.clone(), wide],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("an i64 struct index is not valid LLVM");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::StructIndexNotAConstantI32(t)) if t == "i64"
            ),
            "the error must name the offending type, got: {err}"
        );

        // A register is refused for the same reason: the field has to be known now.
        let reg = Value::from_register("n".to_string(), i32_ty, &mut ctx);

        assert!(
            matches!(
                cursor.build_get_element_ptr(
                    &slot,
                    OperandTy::Inferred,
                    &[zero, reg],
                    None,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::Gep(GepError::StructIndexNotAConstantI32(
                    _
                )))
            ),
            "a non-constant struct index names no field"
        );
    }

    /// A field index past the end names nothing, which `llvm-as` also refuses.
    /// Negative is the same error and not a huge positive one.
    #[test]
    fn a_struct_index_must_be_in_range() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);
        let past_end = Value::from_const(5i32, None, &mut ctx).unwrap();
        let negative = Value::from_const(-1i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_get_element_ptr(
                &slot.clone(),
                OperandTy::Inferred,
                &[zero.clone(), past_end],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("a two-field struct has no field 5");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::StructIndexOutOfRange { index, fields })
                    if *index == 5 && *fields == 2
            ),
            "the error must name the index and the arity, got: {err}"
        );

        assert!(
            matches!(
                cursor.build_get_element_ptr(
                    &slot,
                    OperandTy::Inferred,
                    &[zero, negative],
                    None,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::Gep(GepError::StructIndexOutOfRange {
                    index: -1,
                    ..
                }))
            ),
            "a negative index is out of range, not a large one"
        );
    }

    /// Only aggregates have anything to descend into: a second index against a scalar
    /// names nothing, and `llvm-as` refuses it too.
    #[test]
    fn only_an_aggregate_can_be_indexed_into() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let slot = cursor
            .build_alloca(i32_ty, None, None, "p".into(), &mut ctx)
            .unwrap();

        let zero = value(0, &mut ctx);
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_get_element_ptr(
                &slot,
                OperandTy::Inferred,
                &[zero, one],
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("an i32 has no elements");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::TypeNotIndexable(t)) if t == "i32"
            ),
            "the error must name the type, got: {err}"
        );
    }

    /// `store i64 %v` where `%v` is an `i32` is not a store LLVM will assemble, and
    /// widening it would need a `zext`/`sext` the caller never asked for — so the
    /// declared type has to match what the register actually holds.
    #[test]
    fn a_store_refuses_a_register_that_is_not_the_declared_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let reg = Value::from_register("v".to_string(), i32_ty, &mut ctx);

        let err = cursor
            .build_store(&ptr, &reg, i64_ty.into(), None, &mut ctx)
            .expect_err("an i32 register is not an i64");

        assert!(
            matches!(&err, InstructionError::Store(StoreError::StoredValueTypeMismatch(got, declared))
                if got == "i32" && declared == "i64"),
            "the error must name both types, got: {err}"
        );
    }

    /// A constant, unlike a register, does fold into the declared type — the store
    /// is of the widened constant, so no instruction is skipped.
    #[test]
    fn a_store_widens_a_constant_to_the_declared_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        cursor
            .build_store(&ptr, &seven, i64_ty.into(), None, &mut ctx)
            .expect("an i32 constant stores as an i64");

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::Store(StoreOperands { value, .. }) = &block.instructions[0].kind
        else {
            panic!("expected a store")
        };

        // The *stored* value is the widened one — a store of the original `i32`
        // would be a different instruction than the caller asked for.
        assert_eq!(ctx.display(value.ty()).to_string(), "i64");

        let ValueKind::ConstExpr(ConstExpr::Const(id)) = value.kind() else {
            panic!("expected a constant")
        };

        assert_eq!(*ctx.const_interner.value(id.raw()), ConstValue::I64(7));
    }

    /// `alloca` reserves room for a value, so the type has to have a size. `llvm-as`
    /// refuses `alloca void` ("void type only allowed for function results") and
    /// `alloca` of a function type ("invalid type for alloca") — the same pair
    /// `is_first_class` names, so this gates on the predicate that
    /// `only_void_and_function_types_are_unsized` pins in `value.rs`.
    #[test]
    fn an_alloca_refuses_an_unsized_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let void_ty = ctx.void_ty();

        let err = cursor
            .build_alloca(void_ty, None, None, RegName::Unnamed, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::TypeNotAllocatable(t)) if t == "void"),
            "expected a not-allocatable error, got: {err}"
        );
    }

    /// An aggregate is sized, so it allocates — the predicate is "has a size", not
    /// LLVM's narrower "first class".
    #[test]
    fn an_alloca_accepts_an_aggregate_and_yields_a_pointer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let array_ty = intern(
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        let slot = cursor
            .build_alloca(array_ty, None, None, "buf".into(), &mut ctx)
            .expect("`[4 x i32]` is sized");

        assert_eq!(
            ctx.display(slot.ty()).to_string(),
            "ptr",
            "an alloca yields a pointer, not the allocated type"
        );
    }

    /// `llvm-as` rejects a non-integer element count with "element count must have
    /// integer type", whether it is the value's own type or the one declared for it.
    #[test]
    fn an_alloca_count_must_be_an_integer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let a_float = Value::from_const(1.0f32, None, &mut ctx).unwrap();

        let err = cursor
            .build_alloca(
                i32_ty,
                Some((&a_float, None)),
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("a float is not a count");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::AllocaCountNotAnInteger(t)) if t == "float"),
            "the error must name the offending type, got: {err}"
        );

        // The declared type is checked too, so a count that *is* an integer cannot be
        // relabelled into something that is not.
        let an_int = Value::from_const(4i32, None, &mut ctx).unwrap();

        assert!(
            matches!(
                cursor.build_alloca(
                    i32_ty,
                    Some((&an_int, Some(f64_ty))),
                    None,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::Alloca(
                    AllocaError::AllocaCountNotAnInteger(_)
                ))
            ),
            "`double` is not an integer, whoever supplied it"
        );
    }

    /// `i1` counts as an integer here because it does in LLVM: `alloca i32, i1 %c`
    /// assembles.
    #[test]
    fn an_alloca_count_may_be_an_i1() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let one = Value::from_const(true, None, &mut ctx).unwrap();

        assert!(
            cursor
                .build_alloca(i32_ty, Some((&one, None)), None, RegName::Unnamed, &mut ctx)
                .is_ok(),
            "`alloca i32, i1 %c` is valid LLVM"
        );
    }

    /// A register count is checked, not converted — the same rule as a stored value,
    /// since widening one would need an instruction of its own.
    #[test]
    fn an_alloca_refuses_a_register_count_of_the_wrong_width() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let n = Value::from_register("n".to_string(), i32_ty, &mut ctx);

        let err = cursor
            .build_alloca(
                i32_ty,
                Some((&n, Some(i64_ty))),
                None,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("an i32 register is not an i64 count");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::AllocaCountTypeMismatch(got, declared))
                if got == "i32" && declared == "i64"),
            "the error must name both types, got: {err}"
        );
    }

    /// A register's definition is recorded with the block it was defined in, not just
    /// the instruction index.
    ///
    /// The index is a position in *that block's* instruction list, so on its own it
    /// says nothing — reading it against another block lands on an unrelated
    /// instruction, or past the end when that block is shorter. This is the shape
    /// `try_inferring_pointee_ty` depends on to walk back to an `alloca` in the entry
    /// block from a `store` in a later one.
    #[test]
    fn a_register_definition_records_the_block_it_was_defined_in() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let ptr = Value::from_const(NullPtr, None, &mut ctx).unwrap();

        // Two instructions in `entry`, so the register in `body` gets an index that
        // is only valid against `body` — index 0 of a one-instruction list.
        let in_entry = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();

        in_entry
            .build_load(&ptr.clone(), i32_ty.into(), None, "a".into(), &mut ctx)
            .unwrap();

        let second = in_entry
            .build_load(&ptr.clone(), i64_ty.into(), None, "b".into(), &mut ctx)
            .unwrap();

        let in_body = builder.cursor_at_block(body);

        let third = in_body
            .build_load(&ptr, f64_ty.into(), None, "c".into(), &mut ctx)
            .unwrap();

        let func_id = ctx.get_block(entry).func_id;

        let def_of = |val: &Value, ctx: &Context| {
            let ValueKind::Reg(reg) = val.kind() else {
                panic!("a load defines a register")
            };

            *ctx.register_defs(func_id)
                .get(&reg.name)
                .expect("the definition was recorded")
        };

        let second_def = def_of(&second, &ctx);
        let third_def = def_of(&third, &ctx);

        assert_eq!(second_def.block, entry);
        assert_eq!(second_def.instr_index, 1);
        assert_eq!(third_def.block, body, "`c` belongs to the block it was in");
        assert_eq!(third_def.instr_index, 0, "and `body` numbers from 0 again");

        // The indices collide across blocks, which is exactly why the block has to be
        // stored: `c`'s index resolved against `entry` reads `a`, a different
        // instruction of a different type.
        let loaded_ty_at = |b, index: usize, ctx: &Context| {
            let InstructionKind::Load(LoadOperands { ty, .. }) =
                &ctx.get_block(b).instructions[index].kind
            else {
                panic!("expected a load")
            };

            ctx.display(*ty).to_string()
        };

        assert_eq!(
            loaded_ty_at(third_def.block, third_def.instr_index, &ctx),
            "double"
        );

        assert_eq!(
            loaded_ty_at(entry, third_def.instr_index, &ctx),
            "i32",
            "the same index against the wrong block reads an unrelated instruction"
        );
    }

    /// A load yields a value of the type it *loaded*, not of the pointer it read
    /// through — `%x = load i32, ptr %p` defines an `i32`.
    #[test]
    fn a_load_yields_a_value_of_the_loaded_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let loaded = cursor
            .build_load(&ptr, i32_ty.into(), None, "x".into(), &mut ctx)
            .expect("loading an i32 through a ptr is fine");

        assert_eq!(ctx.ty_interner.value(loaded.ty().raw()), &Type::I32);
    }

    /// The instruction records the register it defines, which is what an emitter
    /// writes in front of it.
    #[test]
    fn a_load_records_the_register_it_defines() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();

        cursor
            .build_load(&ptr, i64_ty.into(), None, "x".into(), &mut ctx)
            .unwrap();

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        assert_eq!(block.instructions.len(), 1);

        assert!(
            block.instructions[0].value.is_some(),
            "a load produces a value, so it defines a register"
        );
    }

    /// Memory is addressed through a pointer. Reaching it from an integer needs an
    /// `inttoptr` first, so a non-pointer operand is refused rather than folded.
    #[test]
    fn a_load_needs_a_pointer_operand() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);
        let not_a_ptr = value(7, &mut ctx);

        let i32_ty = ctx.i32_ty();

        let err = cursor
            .build_load(&not_a_ptr, i32_ty.into(), None, "x".into(), &mut ctx)
            .expect_err("an i32 is not an address");

        assert!(
            matches!(&err, InstructionError::PointerOperandExpected(t) if t == "i32"),
            "the error must name the offending type, got: {err}"
        );

        assert!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.is_empty(),
            "the refused load must not have been added"
        );
    }

    /// `void` has no size, so there is nothing to load. `llvm-as` rejects it with
    /// "void type only allowed for function results".
    #[test]
    fn only_sized_types_can_be_loaded() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        // `Type::Func` is the other unsized case; it is covered in `value.rs`,
        // where a `FuncSignature` can be built.
        let void_ty = ctx.void_ty();

        let err = cursor
            .build_load(&ptr, void_ty.into(), None, RegName::Unnamed, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, InstructionError::TypeNotLoadable(t) if t == "void"),
            "expected a not-loadable error, got: {err}"
        );
    }

    /// Aggregates are loadable even though LLVM does not call them "first class" —
    /// `load {i32, i32}` and `load [4 x i32]` both assemble.
    #[test]
    fn aggregates_can_be_loaded() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = intern(Type::I32, &mut ctx);

        for ty in [
            Type::Struct {
                fields: Box::new([i32_ty, i32_ty]),
                packed: false,
            },
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            Type::Ptr,
            Type::Double,
        ] {
            let id = intern(ty, &mut ctx);
            let spelled = rendered(id, &ctx);

            assert!(
                cursor
                    .build_load(&ptr.clone(), id.into(), None, RegName::Unnamed, &mut ctx)
                    .is_ok(),
                "`{spelled}` is loadable"
            );
        }
    }

    /// An explicit alignment must be a power of two. The cases below are the ones
    /// `llvm-as` accepts and rejects — note `1` is valid (it is `2^0`) and `0` is
    /// not, which an even-number test gets backwards in both directions.
    #[test]
    fn an_explicit_alignment_must_be_a_power_of_two() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        for align in [1, 2, 4, 8, 16, 4096] {
            assert!(
                cursor
                    .build_load(
                        &ptr.clone(),
                        i32_ty.into(),
                        Some(align),
                        RegName::Unnamed,
                        &mut ctx
                    )
                    .is_ok(),
                "align {align} is a power of two"
            );
        }

        for align in [0, 3, 6, 10, 12] {
            let err = cursor
                .build_load(
                    &ptr.clone(),
                    i32_ty.into(),
                    Some(align),
                    RegName::Unnamed,
                    &mut ctx,
                )
                .expect_err("not a power of two");

            assert!(
                matches!(&err, InstructionError::AlignmentNotPowerOfTwo(a) if *a == align),
                "expected an alignment error for {align}, got: {err}"
            );
        }
    }

    /// Omitting the alignment is how the ABI default is asked for, and is always
    /// allowed.
    #[test]
    fn an_omitted_alignment_is_allowed() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        assert!(
            cursor
                .build_load(&ptr, i32_ty.into(), None, RegName::Unnamed, &mut ctx)
                .is_ok()
        );
    }

    /// A block plus the cursor writing into it, for the `icmp` tests below — which
    /// need to read the instruction back out to see what the operands were folded to.
    fn block_for_icmp(ctx: &mut Context, builder: &mut Builder) -> (Cursor, BasicBlockId) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();

        (builder.cursor_at_block(entry), entry)
    }

    /// The constant an `icmp`'s operands were folded to, read back out of the block.
    fn icmp_operand_consts(block: BasicBlockId, ctx: &Context) -> (ConstValue, ConstValue) {
        let instr = ctx
            .blocks
            .get(block.raw())
            .unwrap()
            .instructions
            .last()
            .expect("an instruction was added");

        let InstructionKind::ICmp(operands) = &instr.kind else {
            panic!("the last instruction is not an icmp");
        };

        let read = |v: &Value| match v.kind() {
            ValueKind::ConstExpr(ConstExpr::Const(id)) => *ctx.const_interner.value(id.raw()),
            other => panic!("operand is not a constant: {other:?}"),
        };

        (read(&operands.a), read(&operands.b))
    }

    /// The type an `icmp` settled on for its operands.
    fn icmp_ty(block: BasicBlockId, ctx: &Context) -> TyId {
        let instr = ctx
            .blocks
            .get(block.raw())
            .unwrap()
            .instructions
            .last()
            .expect("an instruction was added");

        let InstructionKind::ICmp(operands) = &instr.kind else {
            panic!("the last instruction is not an icmp");
        };

        operands.ty
    }

    /// An unsigned predicate widens a narrower constant by **zero**-extension, so a
    /// high-bit-set operand keeps its unsigned meaning.
    ///
    /// This is the case that assembled to a wrong answer before `ICond::signedness`
    /// existed: as an unsigned quantity `-1i32` is 4294967295, and `icmp ult` against
    /// it sign-extended flips the result for every operand above 2^32.
    #[test]
    fn an_unsigned_predicate_zero_extends_a_narrower_constant() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, block) = block_for_icmp(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();
        let wide = Value::from_const(5_000_000_000i64, None, &mut ctx).unwrap();
        let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

        let result = cursor
            .build_icmp(
                ICond::Ult,
                OperandTy::Inferred,
                &wide,
                &narrow,
                "c".into(),
                &mut ctx,
            )
            .expect("an i32 constant widens into an i64 comparison");

        assert_eq!(
            icmp_operand_consts(block, &ctx),
            (
                ConstValue::I64(5_000_000_000),
                ConstValue::I64(4_294_967_295)
            ),
            "the narrower operand must be zero-extended, not sign-extended",
        );

        assert!(result.ty.is_i1(&ctx), "an icmp produces an i1");
        assert_eq!(
            icmp_ty(block, &ctx),
            i64_ty,
            "the comparison itself is at the common type",
        );
    }

    /// A signed predicate widens the same constant by **sign**-extension, giving the
    /// other answer — which is why the predicate has to be consulted.
    #[test]
    fn a_signed_predicate_sign_extends_a_narrower_constant() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, block) = block_for_icmp(&mut ctx, &mut builder);

        let wide = Value::from_const(5_000_000_000i64, None, &mut ctx).unwrap();
        let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

        cursor
            .build_icmp(
                ICond::Slt,
                OperandTy::Inferred,
                &wide,
                &narrow,
                "c".into(),
                &mut ctx,
            )
            .expect("an i32 constant widens into an i64 comparison");

        assert_eq!(
            icmp_operand_consts(block, &ctx),
            (ConstValue::I64(5_000_000_000), ConstValue::I64(-1)),
            "the same constant, sign-extended, is a different number",
        );
    }

    /// `eq` and `ne` carry no signedness, so they refuse to widen rather than guess.
    ///
    /// LLVM has one `eq`, because at equal widths the reading cannot change the
    /// answer. It only matters when widening, and there the two choices disagree.
    #[test]
    fn eq_and_ne_refuse_operands_of_different_types() {
        for cond in [ICond::Eq, ICond::Ne] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let wide = Value::from_const(1i64, None, &mut ctx).unwrap();
            let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

            let err = cursor
                .build_icmp(
                    cond,
                    OperandTy::Inferred,
                    &wide,
                    &narrow,
                    RegName::Unnamed,
                    &mut ctx,
                )
                .expect_err("eq/ne must not widen");

            assert!(
                matches!(
                    &err,
                    InstructionError::ICmp(ICmpError::OperandTypesDiffer(p, a, b))
                        if p == &cond.to_string() && a == "i64" && b == "i32"
                ),
                "the error must name the predicate and both types, got: {err}",
            );
        }
    }

    /// The same operands under an *ordered* predicate are accepted, which is what
    /// makes the refusal above about signedness rather than about width.
    #[test]
    fn an_ordered_predicate_accepts_what_eq_refuses() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let wide = Value::from_const(1i64, None, &mut ctx).unwrap();
        let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

        assert!(
            cursor
                .build_icmp(
                    ICond::Ult,
                    OperandTy::Inferred,
                    &wide,
                    &narrow,
                    RegName::Unnamed,
                    &mut ctx
                )
                .is_ok(),
        );
    }

    /// For `eq`/`ne` the type argument is a *check*, not a coercion — there is no
    /// signedness to coerce with, so a type the operands do not have is an error
    /// rather than a request to widen them.
    #[test]
    fn eq_treats_an_explicit_type_as_an_assertion() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();
        let a = Value::from_const(1i32, None, &mut ctx).unwrap();
        let b = Value::from_const(2i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_icmp(ICond::Eq, i64_ty.into(), &a, &b, RegName::Unnamed, &mut ctx)
            .expect_err("i32 operands are not i64, and eq will not widen them");

        assert!(
            matches!(
                &err,
                InstructionError::ICmp(ICmpError::ProvidedTypeDoesNotMatchOperands(p, given, have))
                    if p == "eq" && given == "i64" && have == "i32"
            ),
            "got: {err}",
        );
    }

    /// Matching the operands' own type is accepted, so the check is on the *type*,
    /// not on the argument being absent.
    #[test]
    fn eq_accepts_an_explicit_type_the_operands_already_have() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let a = Value::from_const(1i32, None, &mut ctx).unwrap();
        let b = Value::from_const(2i32, None, &mut ctx).unwrap();

        assert!(
            cursor
                .build_icmp(ICond::Eq, i32_ty.into(), &a, &b, RegName::Unnamed, &mut ctx)
                .is_ok(),
        );
    }

    /// Widening a *register* would need a real `zext`/`sext`, which this builder does
    /// not insert — so a mismatch between registers is refused even under an ordered
    /// predicate, where a constant would have been folded.
    #[test]
    fn a_register_of_the_wrong_width_is_not_widened() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let a = Value::from_register("a".to_string(), i64_ty, &mut ctx);
        let b = Value::from_register("b".to_string(), i32_ty, &mut ctx);

        let err = cursor
            .build_icmp(
                ICond::Ult,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("a register cannot be widened by folding");

        assert!(
            matches!(
                &err,
                InstructionError::ICmp(ICmpError::OperandsNotCastable(p, a, b))
                    if p == "ult" && a == "i64" && b == "i32"
            ),
            "got: {err}",
        );
    }

    /// A constant that does not fit the common type under the predicate's reading is
    /// refused rather than truncated.
    #[test]
    fn a_constant_that_does_not_fit_the_given_type_is_refused() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let i8_ty = ctx.i8_ty();
        let a = Value::from_const(300i32, None, &mut ctx).unwrap();
        let b = Value::from_const(1i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_icmp(ICond::Slt, i8_ty.into(), &a, &b, RegName::Unnamed, &mut ctx)
            .expect_err("300 does not fit an i8");

        assert!(
            matches!(
                &err,
                InstructionError::ICmp(ICmpError::OperandsNotCastable(..))
            ),
            "got: {err}",
        );
    }

    /// `icmp` compares integers or pointers. `llvm-as` refuses `icmp eq float` with
    /// "icmp requires integer operands" — comparing floats needs `fcmp`.
    #[test]
    fn icmp_refuses_floats() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let a = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let b = Value::from_const(2.0f64, None, &mut ctx).unwrap();

        let err = cursor
            .build_icmp(
                ICond::Eq,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("floats are not comparable with icmp");

        assert!(
            matches!(
                &err,
                InstructionError::ICmp(ICmpError::OperandTypeNotComparable(ty)) if ty == "double"
            ),
            "the error must name the offending type, got: {err}",
        );
    }

    /// Pointers are comparable, with *every* predicate. `llvm-as` accepts both
    /// `icmp ult ptr` and `icmp slt ptr`, so the signed ones must not be refused.
    #[test]
    fn icmp_accepts_pointers_under_any_predicate() {
        for cond in [ICond::Eq, ICond::Ult, ICond::Slt] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let a = Value::from_const(NullPtr, None, &mut ctx).unwrap();
            let b = Value::from_const(NullPtr, None, &mut ctx).unwrap();

            assert!(
                cursor
                    .build_icmp(
                        cond,
                        OperandTy::Inferred,
                        &a,
                        &b,
                        RegName::Unnamed,
                        &mut ctx
                    )
                    .is_ok(),
                "`icmp {cond} ptr` is valid LLVM",
            );
        }
    }

    /// `fcmp` widens a narrower float constant, and needs no signedness to do it:
    /// `fpext` is exact, so there is nothing for a caller to choose.
    #[test]
    fn fcmp_widens_a_narrower_float_constant() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, block) = block_for_icmp(&mut ctx, &mut builder);

        let f64_ty = ctx.f64_ty();
        let wide = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let narrow = Value::from_const(0.5f32, None, &mut ctx).unwrap();

        let result = cursor
            .build_fcmp(
                FCond::Olt,
                OperandTy::Inferred,
                &wide,
                &narrow,
                "c".into(),
                &mut ctx,
            )
            .expect("an f32 constant widens into an f64 comparison");

        assert!(result.ty.is_i1(&ctx), "an fcmp produces an i1");

        let instr = ctx
            .blocks
            .get(block.raw())
            .unwrap()
            .instructions
            .last()
            .unwrap();

        let InstructionKind::FCmp(operands) = &instr.kind else {
            panic!("the last instruction is not an fcmp");
        };

        assert_eq!(operands.ty, f64_ty, "the comparison is at the wider type");
        assert_eq!(operands.a.ty(), f64_ty);
        assert_eq!(operands.b.ty(), f64_ty);
    }

    /// Integers are not comparable with `fcmp` — that is what `icmp` is for, and
    /// `llvm-as` keeps the two apart.
    #[test]
    fn fcmp_refuses_integers() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let a = Value::from_const(1i32, None, &mut ctx).unwrap();
        let b = Value::from_const(2i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_fcmp(
                FCond::Oeq,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("integers are not comparable with fcmp");

        assert!(
            matches!(
                &err,
                InstructionError::FCmp(FCmpError::OperandTypeNotFloat(ty)) if ty == "i32"
            ),
            "the error must name the offending type, got: {err}",
        );
    }

    /// An integer paired with a float has no common type under
    /// [`Signedness::NotApplicable`], so the mismatch is caught
    /// as a cast failure before the float check is ever reached.
    #[test]
    fn fcmp_refuses_an_integer_paired_with_a_float() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let a = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let b = Value::from_const(2i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_fcmp(
                FCond::Oeq,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("nothing bridges the integer and float families");

        assert!(
            matches!(
                &err,
                InstructionError::FCmp(FCmpError::OperandsNotCastable(p, a, b))
                    if p == "oeq" && a == "double" && b == "i32"
            ),
            "got: {err}",
        );
    }

    /// Two integers of *different* widths are refused as an uncastable pair, not as
    /// a non-float type.
    ///
    /// This is what [`Signedness::NotApplicable`] buys. Under a
    /// real signedness the narrower constant would widen happily, and the mismatch
    /// would only surface a step later at the float check — reporting the *widened*
    /// type, which the caller never wrote. Refusing the cast keeps the error pointing
    /// at what was actually passed.
    #[test]
    fn fcmp_refuses_two_integers_of_different_widths_as_a_cast_failure() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let a = Value::from_const(1i64, None, &mut ctx).unwrap();
        let b = Value::from_const(2i32, None, &mut ctx).unwrap();

        let err = cursor
            .build_fcmp(
                FCond::Oeq,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx,
            )
            .expect_err("integers are not comparable with fcmp");

        assert!(
            matches!(
                &err,
                InstructionError::FCmp(FCmpError::OperandsNotCastable(p, a, b))
                    if p == "oeq" && a == "i64" && b == "i32"
            ),
            "both original types must be named, not a widened one: {err}",
        );
    }

    /// Widening a *register* would need a real `fpext`, which this builder does not
    /// insert — so a mismatch between registers is refused.
    #[test]
    fn fcmp_does_not_widen_a_register() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();
        let a = Value::from_register("a".to_string(), f64_ty, &mut ctx);
        let b = Value::from_register("b".to_string(), f32_ty, &mut ctx);

        assert!(matches!(
            cursor.build_fcmp(
                FCond::Ogt,
                OperandTy::Inferred,
                &a,
                &b,
                RegName::Unnamed,
                &mut ctx
            ),
            Err(InstructionError::FCmp(FCmpError::OperandsNotCastable(..)))
        ),);
    }

    /// Every predicate assembles, including the two constants — `llvm-as` refuses
    /// `fcmp true` written *without* operands, so they still take two.
    #[test]
    fn every_float_predicate_builds() {
        for cond in [
            FCond::Oeq,
            FCond::Ogt,
            FCond::Oge,
            FCond::Olt,
            FCond::Ole,
            FCond::One,
            FCond::Ord,
            FCond::Ueq,
            FCond::Ugt,
            FCond::Uge,
            FCond::Ult,
            FCond::Ule,
            FCond::Une,
            FCond::Uno,
            FCond::True,
            FCond::False,
        ] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let a = Value::from_const(1.0f64, None, &mut ctx).unwrap();
            let b = Value::from_const(2.0f64, None, &mut ctx).unwrap();

            assert!(
                cursor
                    .build_fcmp(
                        cond,
                        OperandTy::Inferred,
                        &a,
                        &b,
                        RegName::Unnamed,
                        &mut ctx
                    )
                    .is_ok(),
                "`fcmp {cond}` is valid LLVM",
            );
        }
    }

    /// An operation LLVM spells with a signedness widens a narrower constant the way
    /// that spelling says.
    #[test]
    fn a_signed_operation_widens_a_constant_by_its_own_reading() {
        for (op, expected) in [
            (IArithmeticOp::Lshr, ConstValue::I64(4_294_967_295)),
            (IArithmeticOp::Ashr, ConstValue::I64(-1)),
            (IArithmeticOp::Udiv, ConstValue::I64(4_294_967_295)),
            (IArithmeticOp::Sdiv, ConstValue::I64(-1)),
        ] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, block) = block_for_icmp(&mut ctx, &mut builder);

            let wide = Value::from_const(8i64, None, &mut ctx).unwrap();
            let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

            cursor
                .build_iarithmetic(
                    op,
                    OperandTy::Inferred,
                    &wide,
                    &narrow,
                    "r".into(),
                    &mut ctx,
                )
                .expect("a signed operation may widen a constant");

            let instr = ctx
                .blocks
                .get(block.raw())
                .unwrap()
                .instructions
                .last()
                .unwrap();

            let InstructionKind::IArithmetic(operands) = &instr.kind else {
                panic!("not an iarithmetic");
            };

            let ValueKind::ConstExpr(ConstExpr::Const(id)) = operands.b.kind() else {
                panic!("the right operand is not a constant");
            };

            assert_eq!(
                *ctx.const_interner.value(id.raw()),
                expected,
                "`{op}` must widen by its own reading",
            );
        }
    }

    /// The seven operations with no signedness refuse to widen, because nothing says
    /// which way to fill the new bits and the two answers differ.
    ///
    /// `add i64 100, -1` is 99; the same `i32` constant zero-extended gives
    /// 4294967395. LLVM has one `add` precisely because the *result* bits do not
    /// depend on the reading — but the *widening* does.
    #[test]
    fn a_signedness_free_operation_refuses_to_widen() {
        for op in [
            IArithmeticOp::Add,
            IArithmeticOp::Sub,
            IArithmeticOp::Mul,
            IArithmeticOp::Shl,
            IArithmeticOp::And,
            IArithmeticOp::Or,
            IArithmeticOp::Xor,
        ] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let wide = Value::from_const(100i64, None, &mut ctx).unwrap();
            let narrow = Value::from_const(-1i32, None, &mut ctx).unwrap();

            let err = cursor
                .build_iarithmetic(
                    op,
                    OperandTy::Inferred,
                    &wide,
                    &narrow,
                    RegName::Unnamed,
                    &mut ctx,
                )
                .expect_err("no reading is available, so widening must be refused");

            assert!(
                matches!(
                    &err,
                    InstructionError::IArithmetic(IArithmeticError::OperandTypesDiffer(o, a, b))
                        if o == &op.to_string() && a == "i64" && b == "i32"
                ),
                "got: {err}",
            );
        }
    }

    /// Matching operands are fine for those same operations — the refusal is about
    /// widening, not about the operation.
    #[test]
    fn a_signedness_free_operation_accepts_matching_operands() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();
        let a = Value::from_const(100i64, None, &mut ctx).unwrap();
        let b = Value::from_const(-1i64, None, &mut ctx).unwrap();

        let result = cursor
            .build_iarithmetic(
                IArithmeticOp::Add,
                OperandTy::Inferred,
                &a,
                &b,
                "r".into(),
                &mut ctx,
            )
            .expect("two i64s need no widening");

        assert_eq!(
            result.ty(),
            i64_ty,
            "arithmetic yields the operand type, not an i1",
        );
    }

    /// Floats need the `f`-prefixed instructions, and integers the unprefixed ones.
    #[test]
    fn the_integer_and_float_instructions_refuse_each_other() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let f = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let g = Value::from_const(2.0f64, None, &mut ctx).unwrap();

        assert!(
            matches!(
                cursor.build_iarithmetic(
                    IArithmeticOp::Add,
                    OperandTy::Inferred,
                    &f,
                    &g,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::IArithmetic(
                    IArithmeticError::OperandTypeNotInteger(..)
                ))
            ),
            "`add` does not take doubles",
        );

        let i = Value::from_const(1i32, None, &mut ctx).unwrap();
        let j = Value::from_const(2i32, None, &mut ctx).unwrap();

        assert!(
            matches!(
                cursor.build_farithmetic(
                    FArithmeticOp::FAdd,
                    OperandTy::Inferred,
                    &i,
                    &j,
                    RegName::Unnamed,
                    &mut ctx
                ),
                Err(InstructionError::FArithmetic(
                    FArithmeticError::OperandTypeNotFloat(..)
                ))
            ),
            "`fadd` does not take i32s",
        );
    }

    /// `fneg` takes one operand, so it cannot be handed a second — the type system
    /// enforces that, and this pins the type and result.
    #[test]
    fn fneg_is_unary_and_yields_the_operand_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, block) = block_for_icmp(&mut ctx, &mut builder);

        let f64_ty = ctx.f64_ty();
        let v = Value::from_const(1.5f64, None, &mut ctx).unwrap();

        let result = cursor
            .build_fneg(v, "n".into(), &mut ctx)
            .expect("a double may be negated");

        assert_eq!(result.ty(), f64_ty);

        let instr = ctx
            .blocks
            .get(block.raw())
            .unwrap()
            .instructions
            .last()
            .unwrap();

        assert!(
            matches!(&instr.kind, InstructionKind::FNeg(o) if o.ty == f64_ty),
            "the instruction must be an FNeg at the operand type",
        );
    }

    /// `fneg` refuses an integer: there is no integer negation instruction, and
    /// negating one is `sub 0, %x`.
    #[test]
    fn fneg_refuses_an_integer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let v = Value::from_const(1i32, None, &mut ctx).unwrap();

        assert!(matches!(
            cursor.build_fneg(v, RegName::Unnamed, &mut ctx),
            Err(InstructionError::FArithmetic(
                FArithmeticError::OperandTypeNotFloat(..)
            ))
        ),);
    }

    /// Every integer operation builds, so no spelling is unreachable.
    #[test]
    fn every_integer_operation_builds() {
        for op in [
            IArithmeticOp::Add,
            IArithmeticOp::Sub,
            IArithmeticOp::Mul,
            IArithmeticOp::Udiv,
            IArithmeticOp::Sdiv,
            IArithmeticOp::Urem,
            IArithmeticOp::Srem,
            IArithmeticOp::Shl,
            IArithmeticOp::Lshr,
            IArithmeticOp::Ashr,
            IArithmeticOp::And,
            IArithmeticOp::Or,
            IArithmeticOp::Xor,
        ] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let a = Value::from_const(8i32, None, &mut ctx).unwrap();
            let b = Value::from_const(2i32, None, &mut ctx).unwrap();

            assert!(
                cursor
                    .build_iarithmetic(op, OperandTy::Inferred, &a, &b, RegName::Unnamed, &mut ctx)
                    .is_ok(),
                "`{op}` is valid LLVM",
            );
        }
    }

    /// And every float one.
    #[test]
    fn every_float_operation_builds() {
        for op in [
            FArithmeticOp::FAdd,
            FArithmeticOp::FSub,
            FArithmeticOp::FMul,
            FArithmeticOp::FDiv,
            FArithmeticOp::FRem,
        ] {
            let (mut ctx, mut builder) = fixture();
            let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

            let a = Value::from_const(8.0f64, None, &mut ctx).unwrap();
            let b = Value::from_const(2.0f64, None, &mut ctx).unwrap();

            assert!(
                cursor
                    .build_farithmetic(op, OperandTy::Inferred, &a, &b, RegName::Unnamed, &mut ctx)
                    .is_ok(),
                "`{op}` is valid LLVM",
            );
        }
    }

    /// An `icmp` defines an `i1` whatever it compared, and the register it defines is
    /// named from the enclosing function's counter like any other.
    #[test]
    fn an_icmp_defines_an_i1_register() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_for_icmp(&mut ctx, &mut builder);

        let a = Value::from_const(1i64, None, &mut ctx).unwrap();
        let b = Value::from_const(2i64, None, &mut ctx).unwrap();

        let result = cursor
            .build_icmp(
                ICond::Sgt,
                OperandTy::Inferred,
                &a,
                &b,
                "cmp".into(),
                &mut ctx,
            )
            .unwrap();

        assert!(result.ty.is_i1(&ctx));

        let as_value: Value = result.into();

        assert!(matches!(as_value.kind(), ValueKind::Reg(_)));
    }
}
