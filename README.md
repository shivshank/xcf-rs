# XCF File Utility

[![Latest Version](https://img.shields.io/crates/v/xcf.svg)](https://crates.io/crates/xcf)

I wrote this for extracting layer and pixel data from XCF files for my game's art pipeline.

Currently this is its only functionality (and these must either be RGB or RGBA images). Tests
needed. Contributions welcome.

# Example

```rust
extern crate xcf;
use xcf::Xcf;

fn main() {
    let mut rdr = File::open("untitled.xcf")
        .expect("Failed to open file.");
    let raw_image = Xcf::parse(&mut rdr)
        .expect("Failed to parse XCF file.");
}
```
