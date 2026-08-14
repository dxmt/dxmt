// Unit tests harness
//
// This file allows running all unit tests with: cargo test --test unit
//
// Unit tests are fast, isolated tests of individual components.

mod fixtures;
mod common;

// Include unit test modules
mod validation_test {
    include!("unit/validation_test.rs");
}

// Add more unit test modules here as they are created
