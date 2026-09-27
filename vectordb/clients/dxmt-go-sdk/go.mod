// Its own module, not part of the server crate, for the same reason the Python, JavaScript and
// Rust clients are their own packages: a client's dependencies have no business reaching the
// server binary, and a user of one should not have to build the other.
module github.com/dxmt/dxmt/vectordb/clients/dxmt-go-sdk

go 1.18
