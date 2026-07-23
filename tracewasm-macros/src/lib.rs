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
//! name. The macro generates `ImportRegistry::{execute, signature, size}` and,
//! for every tagged method, a compile-time assertion that its signature is a
//! valid `ImportedFunc` (params/results are `WasmTy` tuples).
//!
//! Paths in the generated code are rooted at `::tracewasm_core`, which resolves
//! inside `tracewasm-core` itself via its `extern crate self as tracewasm_core`
//! alias.

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

    for item in &mut impl_block.items {
        let ImplItem::Fn(func) = item else {
            continue;
        };

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

    let mut execute_arms = Vec::new();
    let mut signature_arms = Vec::new();
    let mut asserts = Vec::new();

    for entry in &entries {
        let module = &entry.module;
        let fn_ident = &entry.fn_ident;
        let fn_name = LitStr::new(&fn_ident.to_string(), fn_ident.span());
        let param_types = &entry.param_types;
        let ret_type = &entry.ret_type;
        let indices: Vec<Literal> = (0..param_types.len())
            .map(Literal::usize_unsuffixed)
            .collect();

        // `execute`: decode each argument from the value slice with `WasmTy`,
        // call the method, then encode the result tuple back to `Val`s.
        execute_arms.push(quote! {
            (#module, #fn_name) => {
                let __res = self.#fn_ident(
                    #(
                        <#param_types as ::tracewasm_core::instance::traits::WasmTy>::from_val(params[#indices])
                            .expect("import argument type mismatch (validated at instantiation)")
                    ),*
                );

                ::core::result::Result::Ok(
                    ::tracewasm_core::instance::traits::FuncSignatureEntity::to_vals(&__res)
                        .into_boxed_slice(),
                )
            }
        });

        // `signature`: the declared `(params, results)` value types.
        signature_arms.push(quote! {
            (#module, #fn_name) => ::core::option::Option::Some((
                ::std::vec![
                    #( <#param_types as ::tracewasm_core::instance::traits::WasmTy>::ty() ),*
                ]
                .into_boxed_slice(),
                <#ret_type as ::tracewasm_core::instance::traits::FuncSignatureEntity>::types().into_boxed_slice(),
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
                ::std::boxed::Box<[::tracewasm_core::instance::traits::Val]>,
                ::tracewasm_core::error::TraceWasmError,
            > {
                match (module_name, func_name) {
                    #(#execute_arms)*
                    _ => ::core::result::Result::Err(
                        ::tracewasm_core::error::TraceWasmError::ImportedFunctionNotFound(
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

            fn size(&self) -> u32 {
                #count
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
