use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor};
use std::io::{Read, Write};
use std::path::Path;

use csv;
use serde::de::DeserializeOwned;

use frame::{ColId, Frame};
use hlist::{HCons, HList, HListExt, HNil, Insertable, Transformer};
use {Column, Result};

pub fn read_csv<H, R>(path: impl AsRef<Path>) -> Result<Frame<H>>
where
    H: HList + Insertable,
    H::Product: Transformer,
    <H::Product as Transformer>::Flattened: DeserializeOwned,
{
    let f = File::open(path)?;
    let f = BufReader::new(f);
    read_reader(f)
}

pub fn read_string<H>(data: &str) -> Result<Frame<H>>
where
    H: HList + Insertable,
    H::Product: Transformer,
    <H::Product as Transformer>::Flattened: DeserializeOwned,
{
    let cur = Cursor::new(data);
    read_reader(cur)
}

pub fn read_reader<R, H>(reader: R) -> Result<Frame<H>>
where
    H: HList + Insertable,
    H::Product: Transformer,
    <H::Product as Transformer>::Flattened: DeserializeOwned,
    R: Read,
{
    let mut reader = csv::Reader::from_reader(reader);
    // TODO check header names against frame names?
    let _headers = reader.headers()?.clone();
    let mut frame: Frame<H> = Frame::empty();
    for row in reader.deserialize() {
        let row: <H::Product as Transformer>::Flattened = row?;
        let row = <H::Product as Transformer>::nest(row);
        // Safe because there is no index yet
        unsafe { frame.insert_row(row) };
    }
    Ok(frame)
}

pub trait WriteBuffer: Sized {
    const CHARS_PER_ELEM_HINT: usize = 10;
    fn write_to_buffer(&self) -> (Vec<u8>, Vec<usize>);
}

impl<T: Display> WriteBuffer for Column<T> {
    fn write_to_buffer(&self) -> (Vec<u8>, Vec<usize>) {
        // TODO this could easily? be multithreaded
        let mut buffer = Vec::with_capacity(self.len() * Self::CHARS_PER_ELEM_HINT);
        let mut strixs = Vec::with_capacity(self.len() + 1);
        strixs.push(0);
        for elem in self.iter() {
            if let Some(e) = elem {
                write!(buffer, "{}", e).unwrap();
            }
            strixs.push(buffer.len())
        }
        (buffer, strixs)
    }
}

fn buffer_slices<'a>(buffer: &'a [u8], indices: &'a [usize]) -> impl Iterator<Item = &'a [u8]> {
    (&indices[..indices.len() - 1])
        .iter()
        .zip(&indices[1..])
        .map(move |(&start, &end)| &buffer[start..end])
}

pub trait WriteToBuffer {
    fn write_to_buffer(&self) -> Vec<(Vec<u8>, Vec<usize>)>;
}

impl<Head, Tail> WriteToBuffer for HCons<Head, Tail>
where
    Head: ColId,
    Column<Head::Output>: WriteBuffer,
    Tail: HList + WriteToBuffer,
{
    fn write_to_buffer(&self) -> Vec<(Vec<u8>, Vec<usize>)> {
        let mut ret = self.tail.write_to_buffer();
        ret.push(self.head.write_to_buffer());
        ret
    }
}

impl WriteToBuffer for HNil {
    fn write_to_buffer(&self) -> Vec<(Vec<u8>, Vec<usize>)> {
        Vec::new()
    }
}

impl<H> Frame<H>
where
    H: HList + WriteToBuffer + HListExt,
{
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let w = File::create(path)?;
        let w = BufWriter::new(w);
        self.write_writer(w)
    }

    pub fn write_writer(&self, mut w: impl Write) -> Result<()> {
        let ncols = self.num_cols();
        for (ix, name) in self.names().iter().enumerate() {
            write!(w, "{}", name)?;
            if ix != ncols - 1 {
                w.write_all(&[b','])?;
            }
        }
        w.write_all(&[b'\n'])?;
        let buffers: Vec<_> = self.hlist.write_to_buffer();
        let mut bufslices: Vec<_> = buffers
            .iter()
            .map(|(buf, ixs)| buffer_slices(&buf, &ixs))
            .collect();
        for _rix in 0..self.len() {
            for (cix, col) in bufslices.iter_mut().enumerate() {
                // unwrap guaranteed to succeed
                w.write_all(col.next().unwrap())?;
                if cix != ncols - 1 {
                    w.write_all(&[b','])?;
                }
            }
            w.write_all(&[b'\n'])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame::test_fixtures::*;
    use frame_alias::*;

    // TODO add string escaping
    #[test]
    fn test_write_writer() -> Result<()> {
        let mut w: Vec<u8> = Vec::new();
        let f = quickframe();
        let _ = f.write_writer(&mut w)?;
        let expect = "int_col,float_col,string_col
1,5,this
2,4,is
3,3,the
4,2,words
";
        assert_eq!(expect, String::from_utf8_lossy(&w));
        Ok(())
    }

    #[test]
    fn test_reader_csv() -> Result<()> {
        let expect = quickframe();
        let csv = "int_col,float_col,string_col
1,5,this
2,4,is
3,3,the
4,2,words";
        let frame: Frame3<IntT, FloatT, StringT> = read_string(csv)?;
        assert_eq!(frame, expect);
        Ok(())
    }
}
