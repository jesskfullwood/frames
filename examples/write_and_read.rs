#[macro_use]
extern crate frames;
extern crate tempdir;

use frames::frame::Frame3;
use frames::*;
use tempdir::TempDir;

use std::time;

define_col!(Int, i64, ints);
define_col!(Float, f64, floats);
define_col!(Bool, bool, bools);
define_col!(Strings, String, strings);

fn main() -> Result<()> {
    let size = 10_000_000;
    let df = Frame::new()
        .addcol::<Int, _>(0..size)?
        .addcol::<Float, _>((0..size).map(|v| v as f64 + 0.5))?
        // .addcol::<Bool, _>((0..size).map(|v| v % 2 == 0))?
        .addcol::<Strings, _>((0..size).map(|v| format!("number {}", v)))?;

    let tmp_dir = TempDir::new("framesexample")?;
    let path = tmp_dir.path().join("test.csv");
    println!("Writing CSV to {}", path.display());
    let startw = time::Instant::now();
    df.write_csv(&path)?;
    let durw = time::Instant::now() - startw;
    let tw = durw.as_secs() as f64 + durw.subsec_millis() as f64 / 1000.;
    println!("Wrote in {}s ({:.0} lines/sec)", tw, size as f64 / tw);

    println!("Reading CSV from {}", path.display());
    let startr = time::Instant::now();
    let df2: Frame3<Int, Float, Strings> = read_csv(&path).unwrap();
    let durr = time::Instant::now() - startr;
    let tr = durr.as_secs() as f64 + durr.subsec_millis() as f64 / 1000.;
    println!("Read in {}s ({:.0} lines/sec)", tr, size as f64 / tr);

    // println!("{}", df2);

    // for (name, col) in df.itercols() {
    //     if let Some(d) = col.describe() {
    //         println!("{}: {:?}", name, d)
    //     }
    // }

    Ok(())
}
