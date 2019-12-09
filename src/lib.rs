//! Read pixel data from GIMP's native XCF files.
//!
//! See [`Xcf`] for usage (methods `open` and `load`). For extracting pixel data, you probably want
//! to access a layer via `Xcf::layer` and `Layer::raw_sub_buffer`, which you can use to
//! create `ImageBuffer`s from the `image` crate. You can also do direct pixel access via
//! `Layer::pixel`.
//!
//! [`Xcf`]: struct.Xcf.html

#[macro_use]
extern crate derive_error;
extern crate byteorder;

use byteorder::{ReadBytesExt, BigEndian};

use std::string;
use std::cmp;
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::borrow::Cow;

#[derive(Debug, Error)]
pub enum Error {
    Io(io::Error),
    Utf8(string::FromUtf8Error),
    InvalidFormat,
    UnknownVersion,
    NotSupported,
}

#[derive(Debug)]
pub struct Xcf {
    pub header: XcfHeader,
    pub layers: Vec<Layer>,
}

/// A GIMP XCF file.
///
/// If you need to access multiple layers at once, access layers field and use `split_at`.
impl Xcf {
    /// Open an XCF file at the path specified.
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Xcf, Error> {
        let rdr = BufReader::new(File::open(p)?);
        Xcf::load(rdr)
    }

    /// Read an XCF file from a Reader.
    pub fn load<R: Read + Seek>(mut rdr: R) -> Result<Xcf, Error> {
        let header = XcfHeader::parse(&mut rdr)?;

        let mut layers = Vec::new();
        loop {
            let layer_pointer = rdr.read_u32::<BigEndian>()?;
            if layer_pointer == 0 {
                break;
            }
            let current_pos = rdr.seek(SeekFrom::Current(0))?;
            rdr.seek(SeekFrom::Start(layer_pointer as u64))?;
            layers.push( Layer::parse(&mut rdr)? );
            rdr.seek(SeekFrom::Start(current_pos))?;
        }

        // TODO: Read channels

        Ok(Xcf {
            header, layers,
        })
    }

    /// Get the width of the canvas.
    pub fn width(&self) -> u32 {
        self.header.width
    }

    /// Get the height of the canvas.
    pub fn height(&self) -> u32 {
        self.header.height
    }

    // Get the dimensions (width, height) of the canvas.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    pub fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter()
            .find(|l| l.name == name)
    }

    pub fn layer_mut(&mut self, name: &str) -> Option<&mut Layer> {
        self.layers.iter_mut()
            .find(|l| l.name == name)
    }
}

#[derive(Debug, PartialEq)]
pub struct XcfHeader {
    pub version: Version,
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    pub properties: Vec<Property>,
}

impl XcfHeader {
    fn parse<R: Read>(mut rdr: R) -> Result<XcfHeader, Error> {
        let mut magic = [0u8; 9];
        rdr.read_exact(&mut magic)?;
        if magic != *b"gimp xcf " {
            return Err(Error::InvalidFormat);
        }

        let version = {
            let mut v = [0u8; 4];
            rdr.read_exact(&mut v)?;
            match &v {
                b"file" => Version::V0,
                b"v001" => Version::V1,
                b"v002" => Version::V2,
                b"v003" => Version::V3,
                _ => return Err(Error::UnknownVersion),
            }
        };

        rdr.read_exact(&mut [0u8])?;

        let width = rdr.read_u32::<BigEndian>()?;
        let height = rdr.read_u32::<BigEndian>()?;

        let color_type = ColorType::new(rdr.read_u32::<BigEndian>()?)?;

        if color_type != ColorType::Rgb {
            unimplemented!("Only RGB/RGBA color images supported");
        }

        let properties = Property::parse_list(&mut rdr)?;

        Ok(XcfHeader {
            version, width, height, color_type, properties,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum Version {
    V0,
    V1,
    V2,
    V3,
}

#[repr(u32)]
#[derive(Debug, PartialEq)]
pub enum ColorType {
    Rgb = 0,
    Grayscale = 1,
    Indexed = 2,
}

impl ColorType {
    fn new(kind: u32) -> Result<ColorType, Error> {
        use self::ColorType::*;
        Ok(match kind {
            0 => Rgb,
            1 => Grayscale,
            2 => Indexed,
            _ => return Err(Error::InvalidFormat),
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct Property {
    pub kind: PropertyIdentifier,
    pub length: usize,
    pub payload: PropertyPayload,
}

impl Property {
    // TODO: GIMP usually calculates sizes based on data and goes from that instead of the reported
    // property length... (for known properties)
    fn guess_size(&self) -> usize {
        match self.payload {
            PropertyPayload::ColorMap { colors, .. } => {
                /* apparently due to a GIMP bug sometimes self.length will be n + 4 */
                3 * colors + 4
            },
            // this is the best we can do otherwise
            _ => self.length,
        }
    }

    fn parse<R: Read>(mut rdr: R) -> Result<Property, Error> {
        let kind = PropertyIdentifier::new(rdr.read_u32::<BigEndian>()?);
        let length = rdr.read_u32::<BigEndian>()? as usize;
        let payload = PropertyPayload::parse(&mut rdr, kind, length)?;
        Ok(Property {
            kind, length, payload,
        })
    }

    fn parse_list<R: Read>(mut rdr: R) -> Result<Vec<Property>, Error> {
        let mut props = Vec::new();
        loop {
            let p = Property::parse(&mut rdr)?;
            if let PropertyIdentifier::PropEnd = p.kind {
                break;
            }
            // only push non end
            props.push(p);
        }
        Ok(props)
    }
}

macro_rules! prop_ident_gen {
    (
        pub enum PropertyIdentifier {
            Unknown,
            $(
                $prop:ident = $val:expr
            ),+,
        }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(u32)]
        pub enum PropertyIdentifier {
            $(
                $prop = $val
            ),+,
            // we have to put this at the end, since otherwise it will try to have value zero,
            // we really don't care what it is as long as it doesn't conflict with anything else
            // (however in the macro we have to put it first since it's a parsing issue)
            Unknown,
        }

        impl PropertyIdentifier {
            fn new(prop: u32) -> PropertyIdentifier {
                match prop {
                    $(
                        $val => PropertyIdentifier::$prop
                    ),+,
                    _ => PropertyIdentifier::Unknown,
                }
            }
        }
    }
}

prop_ident_gen! {
    pub enum PropertyIdentifier {
        Unknown,
        PropEnd = 0,
        PropColormap = 1,
        PropOpacity = 6,
        PropVisible = 8,
        PropLinked = 9,
        PropCompression = 17,
        TypeIdentification = 18,
        PropResolution = 19,
        PropTattoo = 20,
        PropParasites = 21,
        PropPaths = 23,
        PropLockContent = 28,
    }
}

#[derive(Debug, PartialEq)]
pub enum PropertyPayload {
    ColorMap {
        colors: usize,
    },
    End,
    Unknown(Vec<u8>),
}

impl PropertyPayload {
    fn parse<R: Read>(mut rdr: R, kind: PropertyIdentifier, length: usize)
            -> Result<PropertyPayload, Error> {
        use self::PropertyIdentifier::*;
        Ok(match kind {
            PropEnd => PropertyPayload::End,
            _ => {
                let mut p = vec![0; length];
                rdr.read_exact(&mut p)?;
                PropertyPayload::Unknown(p)
            }
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct Layer {
    pub width: u32,
    pub height: u32,
    pub kind: LayerColorType,
    pub name: String,
    pub properties: Vec<Property>,
    pub pixels: PixelData,
}

impl Layer {
    fn parse<R: Read + Seek>(mut rdr: R) -> Result<Layer, Error> {
        let width = rdr.read_u32::<BigEndian>()?;
        let height = rdr.read_u32::<BigEndian>()?;
        let kind = LayerColorType::new(rdr.read_u32::<BigEndian>()?)?;
        let name = read_gimp_string(&mut rdr)?;
        let properties = Property::parse_list(&mut rdr)?;
        let hptr = rdr.read_u32::<BigEndian>()?;
        let current_pos = rdr.seek(SeekFrom::Current(0))?;
        rdr.seek(SeekFrom::Start(hptr as u64))?;
        let pixels = PixelData::parse_heirarchy(&mut rdr)?;
        rdr.seek(SeekFrom::Start(current_pos))?;
        // TODO
        // let mptr = rdr.read_u32::<BigEndian>()?;
        Ok(Layer {
            width, height, kind, name, properties, pixels
        })
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<RgbaPixel> {
        self.pixels.pixel(x, y)
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn raw_rgba_buffer(&self) -> Cow<[RgbaPixel]> {
        Cow::from(&self.pixels.pixels)
    }

    pub fn raw_sub_rgba_buffer(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<u8> {
        self.pixels.raw_sub_rgba_buffer(x, y, width, height)
    }
}

#[derive(Debug, PartialEq)]
pub struct LayerColorType {
    pub kind: ColorType,
    pub alpha: bool,
}

impl LayerColorType {
    fn new(identifier: u32) -> Result<LayerColorType, Error> {
        let kind = ColorType::new(identifier / 2)?;
        let alpha = identifier % 2 == 1;
        Ok(LayerColorType {
            alpha, kind,
        })
    }
}

// TODO: Make this an enum? We should store a buffer that matches the channels present.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelData {
    width: usize,
    height: usize,
    pixels: Vec<RgbaPixel>
}

impl PixelData {
    /// Parses the (silly?) heirarchy structure in the xcf file into a pixel array
    /// Makes lots of assumptions! Only supports RGBA for now.
    fn parse_heirarchy<R: Read + Seek>(mut rdr: R) -> Result<PixelData, Error> {
        // read the heirarchy
        let width = rdr.read_u32::<BigEndian>()? as usize;
        let height = rdr.read_u32::<BigEndian>()? as usize;
        let bpp = rdr.read_u32::<BigEndian>()? as usize;
        if bpp != 3 && bpp != 4 {
            return Err(Error::NotSupported);
        }
        let lptr = rdr.read_u32::<BigEndian>()?;
        let _dummpy_ptr_pos = rdr.seek(SeekFrom::Current(0))?;
        rdr.seek(SeekFrom::Start(lptr as u64))?;
        // read the level
        let level_width = rdr.read_u32::<BigEndian>()? as usize;
        let level_height = rdr.read_u32::<BigEndian>()? as usize;
        if level_width != width || level_height != height {
            return Err(Error::InvalidFormat);
        }

        let mut pixels = vec![RgbaPixel([0, 0, 0, 255]); (width * height) as usize];
        let mut next_tptr_pos;

        let tiles_x = (width as f32 / 64.0).ceil() as usize;
        let tiles_y = (height as f32 / 64.0).ceil() as usize;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let tptr = rdr.read_u32::<BigEndian>()?;
                next_tptr_pos = rdr.seek(SeekFrom::Current(0))?;
                rdr.seek(SeekFrom::Start(tptr as u64))?;

                let mut cursor = TileCursor::new(width, height, tx, ty, bpp);
                cursor.feed(&mut rdr, &mut pixels)?;

                rdr.seek(SeekFrom::Start(next_tptr_pos))?;
            }
        }

        // rdr.seek(SeekFrom::Start(dummpy_ptr_pos))?;
        // TODO: dummy levels? do we need to consider them?
        // if we do:
        /*loop {
            let dummy_level_ptr = rdr.read_u32::<BigEndian>()?;
            if dummy_level_ptr == 0 {
                break;
            }
        }*/
        // we are now at the end of the heirarchy structure.

        Ok(PixelData { pixels, width, height })
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<RgbaPixel> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.pixels[y * self.width + x])
    }

    /// Creates a raw sub buffer from self.
    ///
    /// # Panics
    ///
    /// Panics if a pixel access is out of bounds.
    pub fn raw_sub_rgba_buffer(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<u8> {
        let mut sub = Vec::with_capacity(width * height * 4);
        for _y in y..(y + height) {
            for _x in x..(x + width) {
                if _y > self.height || _x > self.width {
                    panic!("Pixel access is out of bounds");
                }
                sub.extend_from_slice(&self.pixel(_x, _y).unwrap().0);
            }
        }
        return sub
    }
}

pub struct TileCursor {
    width: usize,
    height: usize,
    channels: usize,
    x: usize,
    y: usize,
    i: usize,
}

// TODO: I like the use of a struct but this isn't really any kind of cursor.
// The use of a struct allows us to seperate the state we need to refer to from the number of
// stuff we need to store within the "algorithm." A better design is very welcome! i should be
// moved into feed as a local.
impl TileCursor {
    fn new(width: usize, height: usize, tx: usize, ty: usize, channels: usize) -> TileCursor {
        TileCursor {
            width,
            height,
            channels,
            x: tx * 64,
            y: ty * 64,
            i: 0,
        }
    }

    /// Feed the cursor a stream starting at the beginning of an XCF tile structure.
    fn feed<R: Read>(&mut self, mut rdr: R, pixels: &mut [RgbaPixel]) -> Result<(), Error> {
        let twidth = cmp::min(self.x + 64, self.width) - self.x;
        let theight = cmp::min(self.y + 64, self.height) - self.y;
        let base_offset = self.y * self.width + self.x;
        // each channel is laid out one after the other
        let mut channel = 0;
        while channel < self.channels {
            while self.i < twidth * theight {
                let determinant = rdr.read_u8()? as u32;
                if determinant < 127 { // A short run of identical bytes
                    let run = (determinant + 1) as usize;
                    let v = rdr.read_u8()?;
                    for i in (self.i)..(self.i + run) {
                        let index = base_offset + (i / twidth) * self.width + i % twidth;
                        pixels[index].0[channel] = v;
                    }
                    self.i += run;
                } else if determinant == 127 { // A long run of identical bytes
                    let run = rdr.read_u16::<BigEndian>()? as usize;
                    let v = rdr.read_u8()?;
                    for i in (self.i)..(self.i + run) {
                        let index = base_offset + (i / twidth) * self.width + i % twidth;
                        pixels[index].0[channel] = v;
                    }
                    self.i += run;
                } else if determinant == 128 { // A long run of different bytes
                    let stream_run = rdr.read_u16::<BigEndian>()? as usize;
                    for i in (self.i)..(self.i + stream_run) {
                        let index = base_offset + (i / twidth) * self.width + i % twidth;
                        let v = rdr.read_u8()?;
                        pixels[index].0[channel] = v;
                    }
                    self.i += stream_run;
                } else { // A short run of different bytes
                    let stream_run = (256 - determinant) as usize;
                    for i in (self.i)..(self.i + stream_run) {
                        let index = base_offset + (i / twidth) * self.width + i % twidth;
                        let v = rdr.read_u8()?;
                        pixels[index].0[channel] = v;
                    }
                    self.i += stream_run;
                }
            }

            self.i = 0;
            channel += 1;
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbaPixel(pub [u8; 4]);

impl RgbaPixel {
    pub fn r(&self) -> u8 {
        self.0[0]
    }

    pub fn g(&self) -> u8 {
        self.0[1]
    }

    pub fn b(&self) -> u8 {
        self.0[2]
    }

    pub fn a(&self) -> u8 {
        self.0[3]
    }
}

fn read_gimp_string<R: Read>(mut rdr: R) -> Result<String, Error> {
    let length = rdr.read_u32::<BigEndian>()?;
    let mut buffer = vec![0; length as usize - 1];
    rdr.read_exact(&mut buffer)?;
    // read the DUMB trailing null byte... uhh GIMP team RIIR already? ;p
    rdr.read_exact(&mut [0u8])?;
    Ok(String::from_utf8(buffer)?)
}
