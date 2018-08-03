use *;

use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufWriter, Cursor, Read};
use std::path::Path;

enum CollectionBuilder {
    String(Vec<String>),
    Bool(Vec<bool>),
    Float(Vec<Float>),
    Int(Vec<Int>),
}

impl CollectionBuilder {
    fn push(&mut self, record: &str) -> Result<()> {
        use self::CollectionBuilder as CB;
        let out = match self {
            CB::Float(v) => v.push(record.parse::<f64>()?.into()),
            CB::Int(v) => v.push(record.parse::<Int>()?),
            CB::Bool(v) => v.push(record.parse::<bool>()?),
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
        })
        .collect();
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
        } else if item.parse::<f64>().is_ok() {
            CT::Float
        } else if item.parse::<bool>().is_ok() {
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

    // TODO add test for this
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
const CHARS_PER_ELEM: usize = 10;

impl<T: Display> Collection<T> {
    fn write_to_buffer(&self, startix: usize, n_elems: usize) -> (Vec<u8>, Vec<usize>) {
        // TODO this doesn't handle string escaping
        // TODO this could easily? be multithreaded
        let towrite = &self.data[startix..startix + n_elems];
        let mut buffer = Vec::with_capacity(n_elems * CHARS_PER_ELEM);
        let mut strixs = Vec::with_capacity(n_elems + 1);
        strixs.push(0);
        for elem in towrite {
            // TODO is this zero-allocation?? Because it should to be
            write!(buffer, "{}", elem).unwrap();
            strixs.push(buffer.len())
        }
        (buffer, strixs)
    }
}

impl Column {
    fn write_to_buffer(&self, startix: usize, n_elems: usize) -> (Vec<u8>, Vec<usize>) {
        use Column::*;
        match self {
            Int(c) => c.write_to_buffer(startix, n_elems),
            Bool(c) => c.write_to_buffer(startix, n_elems),
            Float(c) => c.write_to_buffer(startix, n_elems),
            String(c) => c.write_to_buffer(startix, n_elems),
        }
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

    // #[test]
    // fn test_basic_write() {
    //     let df = DataFrame::from(
    // }
}
