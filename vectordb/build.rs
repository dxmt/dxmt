//! Codegen and rebuild triggers for the optional interfaces.
//!
//! Everything here is behind a feature. A default build has no protobuf toolchain compiled, no
//! codegen step and no reason to watch the dashboard's build output, which is the same bargain
//! every other optional interface makes.

fn main() {
    // `rust-embed` reads `dashboard/dist/` at compile time. `include_bytes!` makes cargo track
    // the contents of the files it *did* include, which is not the same as tracking the
    // directory: a file added or removed by a rebuild of the bundle leaves the fingerprint
    // unchanged, cargo reports "Finished in 0.34s", and the binary quietly keeps serving the
    // previous build's assets. Observed exactly that way, with a chunk that had been deleted
    // still being served.
    #[cfg(feature = "api-dashboard")]
    println!("cargo:rerun-if-changed=dashboard/dist");

    #[cfg(feature = "api-grpc")]
    {
        println!("cargo:rerun-if-changed=proto/dxmt.proto");
        // `prost-build` shells out to `protoc`. Pointing it at the vendored binary means
        // building this feature does not require protoc to be installed -- one less thing
        // between a contributor and a working build.
        if std::env::var_os("PROTOC").is_none() {
            if let Ok(path) = protoc_bin_vendored::protoc_bin_path() {
                std::env::set_var("PROTOC", path);
            }
        }
        tonic_build::configure()
            .build_client(true)
            .build_server(true)
            .compile(&["proto/dxmt.proto"], &["proto"])
            .expect("compiling proto/dxmt.proto");
    }
}
