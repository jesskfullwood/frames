use *;

use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufWriter, Cursor, Read};
use std::path::Path;

enum CollectionBuilder {
    String(Vec<String>),
    Bool(Vec<Bool>),
    Float(Vec<Float>),
    Int(Vec<Int>),
}

impl CollectionBuilder {
    fn push(&mut self, record: &str) -> Result<()> {
        use self::CollectionBuilder as CB;
        let out = match self {
            CB::Float(v) => v.push(record.parse::<Float>()?.into()),
            CB::Int(v) => v.push(record.parse::<Int>()?),
            CB::Bool(v) => v.push(record.parse::<Bool>()?),
            CB::String(v) => v.push(record.into()),
        };
        Ok(out)
    }

    fn into_column(self) -> Column {
        use self::CollectionBuilder as CB;
        match self {
            CB::Float(v) => Column::from(v),
            CB::Int(v) => Column::from(v),
            CB::Bool(v) => Column::from(v),
            CB::String(v) => Column::from(v),
        }
    }

    fn try_cast(&self, newty: ColType) -> Result<Self> {
        // TODO valid casts:
        // int -> float -> string
        // bool -> string
        use self::CollectionBuilder as CB;
        use ColType as CT;
        let out = match (self, newty) {
            (CB::Int(v), CT::Float) => {
                CB::Float(v.iter().map(|&v| Float::from(v as f64)).collect())
            }
            (CB::Int(v), CT::String) => {
                // TODO could this lose information? Leading/trailing spaces?
                CB::String(v.iter().map(|&v| v.to_string()).collect())
            }
            (CB::Bool(v), CT::String) => CB::String(v.iter().map(|&v| v.to_string()).collect()),
            _ => bail!("Cannot perform conversion"),
        };
        Ok(out)
    }
}

pub fn read_csv(path: impl AsRef<Path>) -> Result<DataFrame> {
    let f = File::open(path)?;
    let f = BufReader::new(f);
    read_reader(f)
}

pub fn read_string(data: &str) -> Result<DataFrame> {
    let cur = Cursor::new(data);
    read_reader(cur)
}

pub fn read_reader<R: Read>(reader: R) -> Result<DataFrame> {
    let mut reader = csv::Reader::from_reader(reader);
    let headers = reader.headers()?.clone();
    let mut csviter = reader.records();
    let row1 = csviter.next().ok_or_else(|| format_err!("No data"))??;
    let mut columns: Vec<_> = row1
        .iter()
        .map(|v| {
            let mut col = ColType::sniff(v).to_builder();
            col.push(v).unwrap();
            col
        }).collect();
    for row in csviter {
        for (elem, col) in row?.iter().zip(columns.iter_mut()) {
            if col.push(elem).is_err() {
                *col = col.try_cast(ColType::sniff(elem))?;
                col.push(elem).unwrap();
            }
        }
    }
    let mut df = DataFrame::new();
    for (name, col) in headers
        .iter()
        .zip(columns.into_iter().map(CollectionBuilder::into_column))
    {
        df.setcol(name, col)?
    }
    Ok(df)
}

impl ColType {
    fn sniff(item: &str) -> ColType {
        use ColType as CT;
        if item.parse::<i64>().is_ok() {
            CT::Int
        } else if item.parse::<Float>().is_ok() {
            CT::Float
        } else if item.parse::<Bool>().is_ok() {
            CT::Bool
        } else {
            CT::String
        }
    }

    fn to_builder(&self) -> CollectionBuilder {
        use ColType as CT;
        match self {
            CT::Int => CollectionBuilder::Int(Vec::new()),
            CT::Float => CollectionBuilder::Float(Vec::new()),
            CT::Bool => CollectionBuilder::Bool(Vec::new()),
            CT::String => CollectionBuilder::String(Vec::new()),
        }
    }
}

impl DataFrame {
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let w = File::create(path)?;
        let w = BufWriter::new(w);
        self.write_writer(w)
    }

    pub fn write_writer(&self, mut w: impl Write) -> Result<()> {
        let ncols = self.num_cols();
        for (ix, name) in self.colnames().iter().enumerate() {
            write!(w, "{}", name)?;
            if ix != ncols - 1 {
                w.write(&[b','])?;
            }
        }
        w.write(&[b'\n'])?;
        let buffers: Vec<_> = self
            .itercols()
            .map(|(_, c)| c.write_to_buffer(0, self.len()))
            .collect();
        let mut bufslices: Vec<_> = buffers
            .iter()
            .map(|(buf, ixs)| buffer_slices(&buf, &ixs))
            .collect();
        for _rix in 0..self.len() {
            for (cix, col) in bufslices.iter_mut().enumerate() {
                // unwrap guaranteed to succeed
                w.write(col.next().unwrap())?;
                if cix != ncols - 1 {
                    w.write(&[b','])?;
                }
            }
            w.write(&[b'\n'])?;
        }
        Ok(())
    }
}

// TODO this could come from a trait

trait WriteBuffer: Sized {
    const CHARS_PER_ELEM_HINT: usize;
    fn write_to_buffer(slice: &[Self]) -> (Vec<u8>, Vec<usize>);
}

// In the absence of specialization, we use this macro
macro_rules! impl_write_buffer {
    ($t:ident, $size_hint:expr) => {
        impl WriteBuffer for $t {
            const CHARS_PER_ELEM_HINT: usize = $size_hint;
            fn write_to_buffer(slice: &[$t]) -> (Vec<u8>, Vec<usize>) {
                // TODO this could easily? be multithreaded
                let mut buffer = Vec::with_capacity(slice.len() * Self::CHARS_PER_ELEM_HINT);
                let mut strixs = Vec::with_capacity(slice.len() + 1);
                strixs.push(0);
                for elem in slice {
                    // TODO is this zero-allocation?? Because it should to be
                    write!(buffer, "{}", elem).unwrap();
                    strixs.push(buffer.len())
                }
                (buffer, strixs)
            }
        }
    };
}

impl_write_buffer!(Float, 6);
impl_write_buffer!(Int, 6);
impl_write_buffer!(Bool, 5);

impl WriteBuffer for String {
    const CHARS_PER_ELEM_HINT: usize = 10;

    fn write_to_buffer(slice: &[String]) -> (Vec<u8>, Vec<usize>) {
        // TODO multithreading
        let mut buffer = Vec::with_capacity(slice.len() * Self::CHARS_PER_ELEM_HINT);
        let mut strixs = Vec::with_capacity(slice.len() + 1);
        strixs.push(0);
        for elem in slice {
            // For convenience we quote everything
            buffer.push(b'"');
            for byte in elem.as_bytes() {
                match byte {
                    b'"' => {
                        // CSV spec says to do this
                        buffer.push(b'"');
                        buffer.push(b'"');
                    }
                    &c => {
                        buffer.push(c);
                    }
                }
            }
            buffer.push(b'"');
            strixs.push(buffer.len())
        }
        (buffer, strixs)
    }
}

impl Column {
    fn write_to_buffer(&self, startix: usize, n_elems: usize) -> (Vec<u8>, Vec<usize>) {
        unimplemented!();
        // use std::string::String;
        // use Column::*;
        // use {Bool, Float, Int};
        // match self {
        //     Int(c) => Int::write_to_buffer(&c.data()[startix..startix + n_elems]),
        //     Float(c) => Float::write_to_buffer(&c.data()[startix..startix + n_elems]),
        //     Bool(c) => Bool::write_to_buffer(&c.data()[startix..startix + n_elems]),
        //     String(c) => String::write_to_buffer(&c.data()[startix..startix + n_elems]),
        // }
    }
}

fn buffer_slices<'a>(buffer: &'a [u8], indices: &'a [usize]) -> impl Iterator<Item = &'a [u8]> {
    (&indices[..indices.len() - 1])
        .iter()
        .zip(&indices[1..])
        .map(move |(&start, &end)| &buffer[start..end])
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_read() {
        let data = "this is,a csv,with 4,cols
1,here,true,1.2
2,are,false,2
0,four,true,0
-4,rows,false,-9";
        let df = read_string(data).unwrap();
        assert_eq!(
            df.coltypes(),
            vec![ColType::Int, ColType::String, ColType::Bool, ColType::Float]
        );
        assert_eq!(df["this is"], Column::from(vec![1, 2, 0, -4]));
    }

    #[test]
    fn test_lookahead_read() {
        use ColType as CT;
        let data = "anum,aword,numword
-5,true,1
0,false,hey
4,False,2
0.6,true,nay
5.,rah,3
";
        let df = read_string(data).unwrap();
        assert_eq!(df.coltypes(), vec![CT::Float, CT::String, CT::String]);
        assert_eq!(df["anum"], Column::from(vec![-5., 0., 4., 0.6, 5.]))
    }

    #[test]
    fn test_basic_write() {
        let words: Vec<String> = "this is some words".split(' ').map(String::from).collect();
        let df = DataFrame::make((
            ("c1", vec![1, 2, 3, 4]),
            ("c2", words),
            ("c3", vec![1., 2., 3., 4.1]),
            ("c4", vec![true, false, false, true]),
        )).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        df.write_writer(&mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let expect = r#"c1,c2,c3,c4
1,"this",1,true
2,"is",2,false
3,"some",3,false
4,"words",4.1,true
"#;
        assert_eq!(expect, out);

        let r = Cursor::new(out);
        let df2 = read_reader(r).unwrap();
        assert_eq!(df, df2);
    }
}

#[test]
fn test_quoted_write() {
    let df = DataFrame::make(((
        "c1",
        vec![String::from(r#"thi,"',s""',"#), String::from("sword")],
    ),)).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    df.write_writer(&mut buf).unwrap();
    let out = String::from_utf8(buf).unwrap();
    let expect = r#"c1
"thi,""',s""""',"
"sword"
"#;
    assert_eq!(expect, out);

    let r = Cursor::new(out);
    let df2 = read_reader(r).unwrap();
    assert_eq!(df, df2);
}
