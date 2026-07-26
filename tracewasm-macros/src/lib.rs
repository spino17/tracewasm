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
//! }
//! ```
//!
//! Each `#[module("...")]`-tagged method becomes an importable function whose
//! module name is the attribute argument and whose field name is the method
//! name. The macro generates
//! `ImportRegistry::{execute, signature, func_count, global_count, get_global}`
//! (where the generated `execute` returns a `ResultVals` wrapper) and, for every
//! tagged method, a compile-time assertion that its signature is a valid
//! `ImportedFunc` (params/results are `WasmTy` tuples).
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
use quote::{format_ident, quote};
use syn::{FnArg, ImplItem, ItemImpl, LitStr, ReturnType, Type, parse_macro_input, parse_quote};

/// A single `#[module("...")]`-tagged host function extracted from the impl.
struct ImportEntry {
    /// The wasm module name from `#[module("...")]`.
    module: LitStr,
    /// The method name, which is also the import's function name.
    fn_ident: syn::Ident,
    /// Parameter types (excluding the `&mut self` receiver), in order.
    param_types: Vec<Type>,
    /// Return type — must implement `Results` (i.e. a tuple like `(i32,)` or `()`).
    ret_type: Type,
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

/// Whether a method takes `&self` (a shared-reference receiver). Imported
/// functions are validated by the `ImportedFunc` trait enforcer, but imported
/// globals have no such enforcer, so the macro requires `&self` explicitly to
/// give a clear error rather than a raw borrow-check failure in `get_global`.
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

        let param_types = func
            .sig
            .inputs
            .iter()
            .filter_map(|input| match input {
                FnArg::Typed(pat_ty) => Some((*pat_ty.ty).clone()),
                FnArg::Receiver(_) => None,
            })
            .collect();

        let ret_type: Type = match &func.sig.output {
            ReturnType::Default => parse_quote!(()),
            ReturnType::Type(_, ty) => (**ty).clone(),
        };

        entries.push(ImportEntry {
            module,
            fn_ident: func.sig.ident.clone(),
            param_types,
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

        // `execute`: decode each argument from the value slice with `WasmTy`,
        // call the method, then marshal the result tuple into `ResultVals`.
        execute_arms.push(quote! {
            (#module, #fn_name) => {
                let __res = self.#fn_ident(
                    #(
                        <#param_types as ::tracewasm_core::instance::traits::WasmTy>::from_val(params[#indices])
                            .expect("import argument type mismatch (validated at instantiation)")
                    ),*
                );

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

        // Compile-time check that the method is a valid `ImportedFunc`: its
        // params form a `Params` tuple and its return type is a `Results` tuple.
        let binds: Vec<syn::Ident> = (0..param_types.len())
            .map(|i| format_ident!("__p{}", i))
            .collect();

        asserts.push(quote! {
            ::tracewasm_core::instance::traits::assert_imported_func_trait(
                |__ctx: &mut #self_ty, ( #(#binds,)* ): ( #(#param_types,)* )| __ctx.#fn_ident( #(#binds),* )
            );
        });
    }

    Ok(quote! {
        #impl_block

        impl ::tracewasm_core::instance::traits::ImportRegistry for #self_ty {
            fn execute(
                &mut self,
                module_name: &str,
                func_name: &str,
                params: &[::tracewasm_core::instance::traits::Val],
            ) -> ::core::result::Result<
                ::tracewasm_core::instance::traits::ResultVals,
                ::tracewasm_core::error::TraceWasmError,
            > {
                match (module_name, func_name) {
                    #(#execute_arms)*
                    _ => ::core::result::Result::Err(
                        ::tracewasm_core::error::TraceWasmError::ImportNotFound(
                            module_name.to_string(),
                            func_name.to_string(),
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
                ::tracewasm_core::error::TraceWasmError
            > {
                match (module_name, global_name) {
                    #(#get_global_arms)*
                    _ => ::core::result::Result::Err(
                        ::tracewasm_core::error::TraceWasmError::ImportNotFound(
                            module_name.to_string(),
                            global_name.to_string(),
                        ),
                    ),
                }
            }
        }

        // Never called; exists only so the closures type-check, asserting each
        // tagged method's signature satisfies `ImportedFunc`.
        const _: () = {
            #[allow(dead_code, non_snake_case, unused_parens, clippy::unused_unit)]
            fn __tracewasm_assert_import_signatures() {
                #(#asserts)*
            }
        };
    })
}
