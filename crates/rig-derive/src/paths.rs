use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn rig_core() -> TokenStream {
    crate_name("rig-core")
        .or_else(|_| crate_name("rig"))
        .map(|found| match found {
            FoundCrate::Itself => quote!(::rig_core),
            FoundCrate::Name(name) => {
                let ident = format_ident!("{name}");
                quote!(::#ident)
            }
        })
        .unwrap_or_else(|_| quote!(::rig))
}
