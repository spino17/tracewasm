use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Error, FnArg, ItemFn, LitStr, ReturnType, parse_macro_input};

/// Registers a free function as a host function. Every module
/// `tracewasm_llvm::jit::JITHandler::parse_module` parses then links it under the
/// function's name, or under `name = "..."` if given. See `tracewasm_llvm::jit::imported`
/// for the full contract.
#[proc_macro_attribute]
pub fn imported(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    match expand(attr.into(), &func) {
        Ok(tokens) => tokens.into(),
        Err(err) => {
            // Keep the function, so the error isn't buried under "not found" ones.
            let mut tokens = err.to_compile_error();

            tokens.extend(quote!(#func));
            tokens.into()
        }
    }
}

fn expand(attr: TokenStream2, func: &ItemFn) -> syn::Result<TokenStream2> {
    let sig = &func.sig;
    let ident = &sig.ident;
    let mut name = None;

    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("name") {
            name = Some(meta.value()?.parse::<LitStr>()?);

            Ok(())
        } else {
            Err(meta.error("expected `name = \"...\"`"))
        }
    });

    syn::parse::Parser::parse2(parser, attr)?;

    let name = name.unwrap_or_else(|| LitStr::new(&ident.to_string(), ident.span()));

    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(Error::new(
            sig.generics.span(),
            "an `#[imported]` function can't be generic: it needs one address",
        ));
    }

    if let Some(asyncness) = sig.asyncness {
        return Err(Error::new(
            asyncness.span(),
            "an `#[imported]` function can't be `async`",
        ));
    }

    if let Some(unsafety) = sig.unsafety {
        return Err(Error::new(
            unsafety.span(),
            "an `#[imported]` function can't be `unsafe`: JIT'd code can't uphold its contract",
        ));
    }

    if let Some(variadic) = &sig.variadic {
        return Err(Error::new(
            variadic.span(),
            "an `#[imported]` function can't be variadic",
        ));
    }

    let mut args = Vec::new();
    let mut tys = Vec::new();

    for (i, input) in sig.inputs.iter().enumerate() {
        match input {
            FnArg::Typed(arg) => {
                args.push(format_ident!("arg{i}"));
                tys.push(&arg.ty);
            }
            FnArg::Receiver(receiver) => {
                return Err(Error::new(
                    receiver.span(),
                    "`#[imported]` only goes on free functions, not methods",
                ));
            }
        }
    }

    let ret = match &sig.output {
        ReturnType::Default => quote!(()),
        ReturnType::Type(_, ty) => quote!(#ty),
    };

    Ok(quote! {
        #func

        const _: () = {
            // The JIT calls through the C ABI, whatever ABI the function itself has.
            // A panic here aborts rather than unwinding into JIT'd code.
            extern "C" fn shim(#(#args: #tys),*) -> #ret {
                #ident(#(#args),*)
            }

            fn link(
                module: &mut ::tracewasm_llvm::jit::JITModule<'_>,
            ) -> ::core::result::Result<(), ::tracewasm_llvm::jit::error::JITError> {
                module.link_host_func(#name, shim as extern "C" fn(#(#tys),*) -> #ret)
            }

            ::tracewasm_llvm::jit::__private::inventory::submit! {
                ::tracewasm_llvm::jit::__private::HostFuncRegistration::new(link)
            }
        };
    })
}
