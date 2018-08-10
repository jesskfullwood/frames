// use *;

// use std::fs::File;
// use std::io::Write;
// use std::io::{BufReader, BufWriter, Cursor, Read};
// use std::path::Path;

// pub fn read_csv(path: impl AsRef<Path>) -> Result<DataFrame> {
//     let f = File::open(path)?;
//     let f = BufReader::new(f);
//     read_reader(f)
// }

// pub fn read_string(data: &str) -> Result<DataFrame> {
//     let cur = Cursor::new(data);
//     read_reader(cur)
// }

// pub fn read_reader<R: Read>(reader: R) -> Result<DataFrame> {
//     let mut reader = csv::Reader::from_reader(reader);
//     let headers = reader.headers()?.clone();
//     let mut csviter = reader.records();
//     let row1 = csviter.next().ok_or_else(|| format_err!("No data"))??;
//     let mut columns: Vec<_> = row1
//         .iter()
//         .map(|v| {
//             let mut col = ColType::sniff(v).to_builder();
//             col.push(v).unwrap();
//             col
//         }).collect();
//     for row in csviter {
//         for (elem, col) in row?.iter().zip(columns.iter_mut()) {
//             if col.push(elem).is_err() {
//                 *col = col.try_cast(ColType::sniff(elem))?;
//                 col.push(elem).unwrap();
//             }
//         }
//     }
//     let mut df = DataFrame::new();
//     for (name, col) in headers
//         .iter()
//         .zip(columns.into_iter().map(CollectionBuilder::into_column))
//     {
//         df.setcol(name, col)?
//     }
//     Ok(df)
// }

// impl ColType {
//     fn sniff(item: &str) -> ColType {
//         use ColType as CT;
//         if item.parse::<i64>().is_ok() {
//             CT::Int
//         } else if item.parse::<Float>().is_ok() {
//             CT::Float
//         } else if item.parse::<Bool>().is_ok() {
//             CT::Bool
//         } else {
//             CT::String
//         }
//     }

//     fn to_builder(self) -> CollectionBuilder {
//         use ColType as CT;
//         match self {
//             CT::Int => CollectionBuilder::Int(Vec::new()),
//             CT::Float => CollectionBuilder::Float(Vec::new()),
//             CT::Bool => CollectionBuilder::Bool(Vec::new()),
//             CT::String => CollectionBuilder::String(Vec::new()),
//         }
//     }
// }

// impl DataFrame {
//     pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
//         let w = File::create(path)?;
//         let w = BufWriter::new(w);
//         self.write_writer(w)
//     }

//     pub fn write_writer(&self, mut w: impl Write) -> Result<()> {
//         let ncols = self.num_cols();
//         for (ix, name) in self.colnames().iter().enumerate() {
//             write!(w, "{}", name)?;
//             if ix != ncols - 1 {
//                 w.write_all(&[b','])?;
//             }
//         }
//         w.write_all(&[b'\n'])?;
//         let buffers: Vec<_> = self
//             .itercols()
//             .map(|(_, c)| c.write_to_buffer(0, self.len()))
//             .collect();
//         let mut bufslices: Vec<_> = buffers
//             .iter()
//             .map(|(buf, ixs)| buffer_slices(&buf, &ixs))
//             .collect();
//         for _rix in 0..self.len() {
//             for (cix, col) in bufslices.iter_mut().enumerate() {
//                 // unwrap guaranteed to succeed
//                 w.write_all(col.next().unwrap())?;
//                 if cix != ncols - 1 {
//                     w.write_all(&[b','])?;
//                 }
//             }
//             w.write_all(&[b'\n'])?;
//         }
//         Ok(())
//     }
// }

// TODO this could come from a trait

// trait WriteBuffer: Sized {
//     const CHARS_PER_ELEM_HINT: usize;
//     fn write_to_buffer(slice: &[Self]) -> (Vec<u8>, Vec<usize>);
// }

// // In the absence of specialization, we use this macro
// macro_rules! impl_write_buffer {
//     ($t:ident, $size_hint:expr) => {
//         impl WriteBuffer for $t {
//             const CHARS_PER_ELEM_HINT: usize = $size_hint;
//             fn write_to_buffer(slice: &[$t]) -> (Vec<u8>, Vec<usize>) {
//                 // TODO this could easily? be multithreaded
//                 let mut buffer = Vec::with_capacity(slice.len() * Self::CHARS_PER_ELEM_HINT);
//                 let mut strixs = Vec::with_capacity(slice.len() + 1);
//                 strixs.push(0);
//                 for elem in slice {
//                     // TODO is this zero-allocation?? Because it should to be
//                     write!(buffer, "{}", elem).unwrap();
//                     strixs.push(buffer.len())
//                 }
//                 (buffer, strixs)
//             }
//         }
//     };
// }

// impl_write_buffer!(Float, 6);
// impl_write_buffer!(Int, 6);
// impl_write_buffer!(Bool, 5);

// impl WriteBuffer for String {
//     const CHARS_PER_ELEM_HINT: usize = 10;

//     fn write_to_buffer(slice: &[String]) -> (Vec<u8>, Vec<usize>) {
//         // TODO multithreading
//         let mut buffer = Vec::with_capacity(slice.len() * Self::CHARS_PER_ELEM_HINT);
//         let mut strixs = Vec::with_capacity(slice.len() + 1);
//         strixs.push(0);
//         for elem in slice {
//             // For convenience we quote everything
//             buffer.push(b'"');
//             for byte in elem.as_bytes() {
//                 match byte {
//                     b'"' => {
//                         // CSV spec says to do this
//                         buffer.push(b'"');
//                         buffer.push(b'"');
//                     }
//                     &c => {
//                         buffer.push(c);
//                     }
//                 }
//             }
//             buffer.push(b'"');
//             strixs.push(buffer.len())
//         }
//         (buffer, strixs)
//     }
// }

// fn buffer_slices<'a>(buffer: &'a [u8], indices: &'a [usize]) -> impl Iterator<Item = &'a [u8]> {
//     (&indices[..indices.len() - 1])
//         .iter()
//         .zip(&indices[1..])
//         .map(move |(&start, &end)| &buffer[start..end])
// }
