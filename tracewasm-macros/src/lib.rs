//! Procedural macros for TraceWasm.
//!
//! The [`macro@imports`] attribute turns an `impl` block of host functions into
//! a generated [`ImportRegistry`] implementation, so embedders only write the
//! function bodies.
//!
//! ```ignore
//! struct HostState { /* ... */ }
//!
//! #[imports]
//! impl HostState {
//!     #[module("env")]
//!     fn add(&mut self, a: i32, b: i32) -> (i32,) {
//!         (a + b,)
//!     }
//!
//!     // A trailing `&mut V` gives the host access to the guest's linear memory.
//!     #[module("env")]
//!     fn store_byte<V: MemoryView>(&mut self, addr: i32, b: i32, mem: &mut V) -> () {
//!         let _ = mem.write_u8(addr as usize, b as u8);
//!     }
//! }
//! ```
//!
//! Each `#[module("...")]`-tagged method becomes an importable function whose
//! module name is the attribute argument and whose field name is the method
//! name. The macro generates
//! `ImportRegistry::{execute, signature, func_count, global_count, get_global}`
//! (where the generated `execute` returns a `ResultVals` wrapper) and, for every
//! tagged method, a compile-time assertion that its parameters form a `Params`
//! tuple and its return type a `Results` tuple.
//!
//! ## The memory view
//!
//! A method may end with a `&mut V` parameter (where `V: MemoryView`) to read and
//! write the instance's linear memory. It is **optional and must come last**: the
//! macro strips it from the wasm signature and forwards the caller's memory after
//! the decoded arguments, so a host function that ignores guest memory can simply
//! leave it off. A `&mut` parameter anywhere else is rejected, since it would
//! shift the wasm arguments.
//!
//! The bound is [`MemoryView`], not [`Memory`]: the generated `execute` is
//! `fn execute<V: MemoryView>(..)`, so a method bounded on the wider `Memory`
//! does not type-check against it. That is the intended capability, not a
//! limitation — `MemoryView` reads and writes but cannot resize, and only the
//! interpreter knows the module's declared maximum and the instance's configured
//! cap that a resize must respect.
//!
//! ## Trapping
//!
//! A method may return `Result<T, E>` instead of `T` to signal a wasm trap. The
//! wasm signature is taken from `T`, and the error is propagated with `?`, so any
//! `E` convertible into an `anyhow::Error` works:
//!
//! ```ignore
//! #[module("env")]
//! fn read_byte<V: MemoryView>(&mut self, addr: i32, mem: &mut V)
//!     -> Result<(i32,), anyhow::Error>
//! {
//!     Ok((mem.read_u8(addr as usize)? as i32,))   // OOB pointer → trap
//! }
//! ```
//!
//! This matters for any host function that touches guest memory: the guest picks
//! the pointer, so an out-of-bounds access is reachable input, not a bug. Without
//! a `Result` the only options are to panic the embedder or fabricate a value.
//!
//! A `#[global("...")]`-tagged method instead declares an importable global: it
//! takes `&self`, returns a single [`WasmTy`] value, and its method name is the
//! global name. The macro routes it through `ImportRegistry::get_global` and
//! counts it in `ImportRegistry::global_count`.
//!
//! ```ignore
//! #[imports]
//! impl HostState {
//!     #[global("env")]
//!     fn table_size(&self) -> i32 {
//!         self.table_size
//!     }
//! }
//! ```
//!
//! Paths in the generated code are absolute (`::tracewasm_core`), so it compiles
//! from any embedder crate that depends on `tracewasm-core`.

use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::quote;
use syn::{FnArg, ImplItem, ItemImpl, LitStr, ReturnType, Type, parse_macro_input, parse_quote};

/// A single `#[module("...")]`-tagged host function extracted from the impl.
struct ImportEntry {
    /// The wasm module name from `#[module("...")]`.
    module: LitStr,
    /// The method name, which is also the import's function name.
    fn_ident: syn::Ident,
    /// The wasm parameter types, in order — the receiver and any trailing
    /// memory-view parameter are excluded, since neither is a wasm value.
    param_types: Vec<Type>,
    /// Whether the method ends with a memory-view parameter (`&mut M`), which the
    /// generated `execute` forwards after the decoded wasm arguments.
    takes_memory_view: bool,
    /// Whether the method returns a `Result`, in which case the generated
    /// `execute` propagates the error with `?` so the host can trap.
    is_fallible: bool,
    /// The success type — must implement `Results` (a tuple like `(i32,)` or
    /// `()`). For a fallible method this is the `T` of its `Result<T, _>`, not the
    /// `Result` itself.
    ret_type: Type,
}

/// If `ty` is a `Result<T, _>`, returns its `T`.
///
/// Matched on the path's last segment, so `Result`, `std::result::Result`, and
/// aliases spelled `…::Result` are all recognised.
fn result_ok_type(ty: &Type) -> Option<Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    let segment = type_path.path.segments.last()?;

    if segment.ident != "Result" {
        return None;
    }

    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };

    args.args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    })
}

/// Whether `ty` is a mutable reference, i.e. the memory-view parameter.
///
/// Unambiguous as a marker: every wasm value type maps to a `Copy` scalar, so a
/// `&mut _` parameter can only be the memory view.
fn is_memory_view(ty: &Type) -> bool {
    matches!(ty, Type::Reference(r) if r.mutability.is_some())
}

/// A single `#[global("...")]`-tagged host global extracted from the impl.
struct GlobalEntry {
    /// The wasm module name from `#[global("...")]`.
    module: LitStr,
    /// The method name, which is also the global's name.
    fn_ident: syn::Ident,
    /// The value type produced — must implement `WasmTy`.
    ret_type: Type,
}

/// Whether a method takes `&self` (a shared-reference receiver).
///
/// The generated `get_global` takes `&self`, so a `#[global("...")]` getter
/// declared any other way cannot be called from it. Rejecting the receiver here
/// puts the error on the method the author wrote, instead of surfacing it as a
/// borrow-check failure inside macro-generated code they cannot see. Imported
/// functions need no equivalent check: `execute` takes `&mut self`, which
/// accommodates every receiver shape.
fn takes_shared_ref(func: &syn::ImplItemFn) -> bool {
    matches!(
        func.sig.inputs.first(),
        Some(FnArg::Receiver(recv)) if recv.reference.is_some() && recv.mutability.is_none()
    )
}

/// Attribute macro placed on an `impl` block: generates an `ImportRegistry`
/// implementation from the `#[module("...")]`-tagged methods. See the crate
/// docs for the expected shape.
#[proc_macro_attribute]
pub fn imports(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);

    match expand(&mut impl_block) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand(impl_block: &mut ItemImpl) -> syn::Result<TokenStream2> {
    let self_ty = impl_block.self_ty.clone();

    // The generated `ImportRegistry` impl repeats the annotated block's generics and
    // where-clause, so a registry with type parameters — `impl<T> Host<T>` — is
    // implemented for the same parameters rather than for an unbound `Host<T>`.
    let generics = impl_block.generics.clone();
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    // Collect the tagged methods, stripping the `#[module(...)]` attribute from
    // each so the impl block can be re-emitted with the user's bodies intact.
    let mut entries: Vec<ImportEntry> = Vec::new();
    let mut globals: Vec<GlobalEntry> = Vec::new();

    for item in &mut impl_block.items {
        let ImplItem::Fn(func) = item else {
            continue;
        };

        // A `#[global("...")]`-tagged method declares an imported global rather
        // than a function; it takes `&self` and returns a single `WasmTy` value.
        if let Some(pos) = func.attrs.iter().position(|a| a.path().is_ident("global")) {
            let module: LitStr = func.attrs.remove(pos).parse_args()?;

            if !takes_shared_ref(func) {
                return Err(syn::Error::new_spanned(
                    &func.sig,
                    "a `#[global(\"...\")]` method must take `&self`",
                ));
            }

            let ret_type: Type = match &func.sig.output {
                ReturnType::Default => {
                    return Err(syn::Error::new_spanned(
                        &func.sig,
                        "a `#[global(\"...\")]` method must return a single `WasmTy` value",
                    ));
                }
                ReturnType::Type(_, ty) => (**ty).clone(),
            };

            globals.push(GlobalEntry {
                module,
                fn_ident: func.sig.ident.clone(),
                ret_type,
            });

            continue;
        }

        let Some(pos) = func.attrs.iter().position(|a| a.path().is_ident("module")) else {
            continue;
        };

        let module: LitStr = func.attrs.remove(pos).parse_args()?;

        let mut param_types: Vec<Type> = func
            .sig
            .inputs
            .iter()
            .filter_map(|input| match input {
                FnArg::Typed(pat_ty) => Some((*pat_ty.ty).clone()),
                FnArg::Receiver(_) => None,
            })
            .collect();

        // A trailing `&mut M` is the memory view, not a wasm parameter: drop it
        // from the signature and forward it separately. It is optional, so a host
        // function that does not touch guest memory can leave it off.
        let takes_memory_view = param_types.last().is_some_and(is_memory_view);

        if takes_memory_view {
            param_types.pop();
        }

        // Only the last parameter may be the memory view; one in the middle would
        // silently shift the wasm arguments.
        if let Some(pos) = param_types.iter().position(is_memory_view) {
            return Err(syn::Error::new_spanned(
                &param_types[pos],
                "the memory-view parameter must come last, after the wasm arguments",
            ));
        }

        let declared_ret: Type = match &func.sig.output {
            ReturnType::Default => parse_quote!(()),
            ReturnType::Type(_, ty) => (**ty).clone(),
        };

        // A `Result` return lets the host trap — reading guest memory can fail on
        // a bad pointer, and that must reach the interpreter rather than panic.
        // The signature is built from the success type; the error is propagated.
        let (is_fallible, ret_type) = match result_ok_type(&declared_ret) {
            Some(ok_ty) => (true, ok_ty),
            None => (false, declared_ret),
        };

        entries.push(ImportEntry {
            module,
            fn_ident: func.sig.ident.clone(),
            param_types,
            takes_memory_view,
            is_fallible,
            ret_type,
        });
    }

    let count = entries.len() as u32;
    let global_count = globals.len() as u32;

    let mut execute_arms = Vec::new();
    let mut signature_arms = Vec::new();
    let mut asserts = Vec::new();
    let mut get_global_arms = Vec::new();

    for global in &globals {
        let module = &global.module;
        let fn_ident = &global.fn_ident;
        let global_name = LitStr::new(&fn_ident.to_string(), fn_ident.span());
        let ret_type = &global.ret_type;

        // `get_global`: call the getter and marshal its value into a `Val`. The
        // `WasmTy::to_val` call also enforces that `ret_type: WasmTy`.
        get_global_arms.push(quote! {
            (#module, #global_name) => ::core::result::Result::Ok(
                <#ret_type as ::tracewasm_core::instance::traits::WasmTy>::to_val(&self.#fn_ident())
            ),
        });
    }

    for entry in &entries {
        let module = &entry.module;
        let fn_ident = &entry.fn_ident;
        let fn_name = LitStr::new(&fn_ident.to_string(), fn_ident.span());
        let param_types = &entry.param_types;
        let ret_type = &entry.ret_type;
        let indices: Vec<Literal> = (0..param_types.len())
            .map(Literal::usize_unsuffixed)
            .collect();

        // A tuple implements `FuncSignatureEntity` at both the param (`[_; 5]`) and
        // result (`[_; 3]`) sizes, so an unqualified `to_vals`/`types` call is
        // ambiguous. Pin the fully-qualified param/result instantiations so the
        // right wrapper types (`ParamVals`/`ParamValTypes`, `ResultVals`/
        // `ResultValTypes`) are selected.
        let params_tuple = quote! { ( #(#param_types,)* ) };
        let param_entity = quote! {
            <#params_tuple as ::tracewasm_core::instance::traits::FuncSignatureEntity<
                [::tracewasm_core::instance::traits::Val; 5],
                [::tracewasm_core::module::ValType; 5],
                ::tracewasm_core::instance::traits::ParamVals,
                ::tracewasm_core::instance::traits::ParamValTypes,
            >>
        };
        let result_entity = quote! {
            <#ret_type as ::tracewasm_core::instance::traits::FuncSignatureEntity<
                [::tracewasm_core::instance::traits::Val; 3],
                [::tracewasm_core::module::ValType; 3],
                ::tracewasm_core::instance::traits::ResultVals,
                ::tracewasm_core::instance::traits::ResultValTypes,
            >>
        };

        // The call's arguments: each wasm parameter decoded from the value slice,
        // then the memory view for the methods that declared one. Collected into one
        // list so the commas come from a single repetition — a method taking the view
        // and no wasm parameters has `memory_view` as its only argument.
        let mut call_args: Vec<proc_macro2::TokenStream> = param_types
            .iter()
            .zip(&indices)
            .map(|(ty, index)| {
                quote! {
                    <#ty as ::tracewasm_core::instance::traits::WasmTy>::from_val(params[#index])
                        .expect("import argument type mismatch (validated at instantiation)")
                }
            })
            .collect();

        if entry.takes_memory_view {
            call_args.push(quote! { memory_view });
        }

        // Propagated with `?`, so any error type the host uses is converted into a
        // `TraceWasmError` through the usual `From` impl.
        let propagate = if entry.is_fallible {
            quote! { ? }
        } else {
            quote! {}
        };

        // `execute`: decode each argument from the value slice with `WasmTy`,
        // call the method, then marshal the result tuple into `ResultVals`.
        execute_arms.push(quote! {
            (#module, #fn_name) => {
                let __res = self.#fn_ident( #(#call_args),* ) #propagate;

                ::core::result::Result::Ok(#result_entity::to_vals(&__res))
            }
        });

        // `signature`: the declared `(params, results)` value types, as the
        // stack-allocated `ParamValTypes`/`ResultValTypes` wrappers.
        signature_arms.push(quote! {
            (#module, #fn_name) => ::core::option::Option::Some((
                #param_entity::types(),
                #result_entity::types(),
            )),
        });

        // Compile-time check that the declared signature is expressible in wasm:
        // the parameters form a `Params` tuple and the return type a `Results`
        // tuple.
        //
        // Asserted on the types directly rather than by coercing the method into
        // a bounded `Fn`: a method taking the memory view is generic over
        // `V: MemoryView`, so it has no single concrete `Fn` shape to check
        // against.
        asserts.push(quote! {
            {
                fn __assert_params<T: ::tracewasm_core::instance::traits::Params>() {}
                fn __assert_results<T: ::tracewasm_core::instance::traits::Results>() {}

                __assert_params::<#params_tuple>();
                __assert_results::<#ret_type>();
            }
        });
    }

    Ok(quote! {
        #impl_block

        impl #impl_generics ::tracewasm_core::instance::traits::ImportRegistry
            for #self_ty #where_clause
        {
            fn execute<V: ::tracewasm_core::memory::MemoryView>(
                &mut self,
                module_name: &str,
                func_name: &str,
                params: &[::tracewasm_core::instance::traits::Val],
                memory_view: &mut V,
            ) -> ::core::result::Result<
                ::tracewasm_core::instance::traits::ResultVals,
                ::tracewasm_core::anyhow::Error,
            > {
                match (module_name, func_name) {
                    #(#execute_arms)*
                    _ => ::core::result::Result::Err(
                        ::tracewasm_core::anyhow::anyhow!(
                            "import not found: {}::{}",
                            module_name,
                            func_name
                        ),
                    ),
                }
            }

            fn signature(
                &self,
                module_name: &str,
                func_name: &str,
            ) -> ::core::option::Option<::tracewasm_core::instance::traits::ImportSignature> {
                match (module_name, func_name) {
                    #(#signature_arms)*
                    _ => ::core::option::Option::None,
                }
            }

            fn func_count(&self) -> u32 {
                #count
            }

            fn global_count(&self) -> u32 {
                #global_count
            }

            fn get_global(
                &self,
                module_name: &str,
                global_name: &str,
            ) -> ::core::result::Result<
                ::tracewasm_core::instance::traits::Val,
                ::tracewasm_core::anyhow::Error
            > {
                match (module_name, global_name) {
                    #(#get_global_arms)*
                    _ => ::core::result::Result::Err(
                        ::tracewasm_core::anyhow::anyhow!(
                            "import not found: {}::{}",
                            module_name,
                            global_name
                        ),
                    ),
                }
            }
        }

        // Never called; type-checking its body is the entire point. Each block
        // instantiates `__assert_params`/`__assert_results` at one tagged
        // method's parameter tuple and return type, so a signature wasm cannot
        // express is rejected here — at the `#[imports]` impl — rather than
        // wherever the generated `execute` or `signature` is first used.
        //
        // Wrapped in a `const _` so the helper is scoped to the expansion and
        // cannot collide with anything the embedder declares.
        const _: () = {
            #[allow(dead_code, non_snake_case, unused_parens, clippy::unused_unit)]
            fn __tracewasm_assert_import_signatures() {
                #(#asserts)*
            }
        };
    })
}

/// Derives a fieldless `…Kind` twin of an enum, plus the mapping from the enum to
/// it.
///
/// For `enum RegInstruction { I32Add(Signature<2, 1>), End, .. }` this generates
/// `enum RegInstructionKind { I32Add, End, .. }`, a `RegInstructionKind::ALL`
/// listing every kind, and `RegInstruction::kind()`.
///
/// # Why
///
/// Rust cannot enumerate the variants of an enum whose variants carry data — a
/// test that wants to visit every instruction has to build one of each by hand,
/// and that list goes stale silently the moment a variant is added. Stripping the
/// fields removes the obstacle: the kinds are unit variants, so `ALL` is just a
/// list of names the macro already has.
///
/// The point is what that buys at the use site. A table keyed by kind is an
/// exhaustive `match`, so adding a variant to the instruction enum stops the
/// table compiling until it is handled, *and* `ALL` grows to include it, so
/// whatever the table says about it actually runs. Neither half can be forgotten,
/// which is what a hand-maintained list of variants could never promise.
#[proc_macro_derive(Kind)]
pub fn kind(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);

    match expand_kind(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_kind(input: &syn::DeriveInput) -> syn::Result<TokenStream2> {
    let syn::Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "`Kind` can only be derived for an enum",
        ));
    };

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "`Kind` does not support generic enums",
        ));
    }

    let name = &input.ident;
    let kind_name = syn::Ident::new(&format!("{name}Kind"), name.span());
    let vis = &input.vis;
    let variants: Vec<_> = data.variants.iter().map(|variant| &variant.ident).collect();

    // One arm per variant, ignoring whatever it carries: `Named { .. }` also
    // matches a tuple variant and a unit one, so a single pattern shape covers
    // all three.
    let arms = variants.iter().map(|variant| {
        quote! { #name::#variant { .. } => #kind_name::#variant }
    });

    let doc = format!("The variants of [`{name}`], without their operands.");

    Ok(quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #vis enum #kind_name {
            #(#variants,)*
        }

        impl #kind_name {
            /// Every kind, in declaration order.
            #vis const ALL: &'static [#kind_name] = &[
                #(#kind_name::#variants,)*
            ];
        }

        impl #name {
            /// Which variant this is, without its operands.
            #vis fn kind(&self) -> #kind_name {
                match self {
                    #(#arms,)*
                }
            }
        }
    })
}
