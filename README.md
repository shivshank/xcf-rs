# XCF File Utility

I built this for extracting layer and pixel data from XCF files for my game's art pipeline.

Currently this is its only functionality (and these must either be RGB or RGBA images). Tests needed. Will most likely break with newer file versions. Contributions welcome.

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