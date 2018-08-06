extern crate heck;
extern crate proc_macro;
extern crate proc_macro2;
#[macro_use]
extern crate quote;
extern crate syn;

use heck::{CamelCase, SnakeCase};
use syn::*;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};

fn ident(s: &str) -> Ident {
    Ident::new(s, Span::call_site())
}

#[proc_macro_derive(Frame)]
pub fn derive(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).expect("derive(Frame) can only be applied to structs");
    let structid = input.ident;
    let struct_data = match input.data {
        Data::Struct(ds) => ds,
        _ => panic!("derive(Frame) can only be applied to struct")
    };
    let fields: Vec<_> = match struct_data.fields {
        Fields::Named(fields) => {
            fields.named.iter().cloned().collect()
        }
        _ => panic!("derive(Frame) may only be applied to named fields")
    };
    let typealias = ident(&format!("{}Frame", structid));
    let framealias = ident(&format!("Frame{}", fields.len()));
    let frame_types: Vec<Ident> = fields.iter().map(|field| {
        let id = field.ident.as_ref().expect("Unnamed field").to_string();
        let cc = id.to_camel_case();
        ident(&cc)
    }).collect();
    let frame_outtypes: Vec<_> = fields.iter().map(|field| {
        field.ty.clone()
    }).collect();
    let modname = ident(&format!("{}", structid).to_snake_case());
    let frame_types1 = frame_types.clone();
    let frame_types2 = frame_types.clone();
    let frame_types3 = frame_types.clone();
    let codegen = quote! {
        mod #modname {
            #(
                struct #frame_types1;
                impl ::frames::frame::ColId for #frame_types2 {
                    type Output = #frame_outtypes;
                }
            )*
            type #typealias = ::frames::frame::#framealias<#(#frame_types3),*>;
        }
    };
    codegen.into()
}
