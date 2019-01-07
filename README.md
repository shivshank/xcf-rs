# XCF File Reader

[![Latest Version](https://img.shields.io/crates/v/xcf.svg)](https://crates.io/crates/xcf)

I wrote this for extracting layer and pixel data from XCF files for my game's art pipeline.

This library has a few notable limitations presently:
 - it always returns RGBA pixels; I have updated the methods to make this clear
    - this can also be fixed! I'll probably get around to it eventually.
 - it can only handle RGB or RGBA images, not grayscale or indexed, with or without alpha.

Tests needed. Contributions welcome.

# Example

```rust
extern crate xcf;
use xcf::Xcf;

fn main() {
    let mut rdr = File::open("untitled.xcf")
        .expect("Failed to open file.");
    let raw_image = Xcf::load(&mut rdr)
        .expect("Failed to parse XCF file.");

    // or simpler yet:
    let raw_image = Xcf::open("untitled.xcf")
        .expect("Failed to open and parse XCF file.");
}
```
