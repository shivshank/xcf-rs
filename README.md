# XCF File Reader

[![Latest Version](https://img.shields.io/crates/v/xcf.svg)](https://crates.io/crates/xcf)

Designed for extracting layer and pixel data from XCF files. 

I originally made this as part of an art pipeline for a game idea, as such it's missing 
support for a lot of features (I only needed to pluck pixel data from a few layers).

 - results are always returned in RGBA pixels, regardless of original format
 - supports RGB or RGBA images, but not grayscale or indexed
 - XCF files with better compression are not supported (there is an ignored failing test 
   for this, should someone like to add support)

Contributions welcome.

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
