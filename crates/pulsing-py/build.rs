//! ``extension-module`` (wheel / maturin) and ``embedded`` (pulsing-cli) must not
//! be enabled on the same ``pulsing-py`` build — PyO3 modes are incompatible.

fn main() {
    let extension = std::env::var("CARGO_FEATURE_EXTENSION_MODULE").is_ok();
    let embedded = std::env::var("CARGO_FEATURE_EMBEDDED").is_ok();
    if extension && embedded {
        panic!(
            "pulsing-py: enable only one of `extension-module` (maturin wheel) \
             or `embedded` (pulsing-cli binary), not both"
        );
    }
}
