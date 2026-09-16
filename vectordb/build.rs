//! Generates the gRPC bindings, and only when they are asked for.
//!
//! Everything here is behind `api-grpc`. A default build has no protobuf toolchain compiled and
//! no codegen step, which is the same bargain every other optional interface makes.

fn main() {
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
