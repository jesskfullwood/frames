#[macro_use]
extern crate frames;
extern crate tempdir;

// use frames::frame::Frame3;
use frames::*;
// use tempdir::TempDir;

use std::time;

define_col!(Int, i64, ints);
define_col!(Float, f64, floats);
define_col!(Bool, bool, bools);
define_col!(Strings, String, strings);

fn main() -> Result<()> {
    let size = 10_000_000;
    let df1 = Frame::new()
        .addcol::<Int, _>((0..size).map(Some))?
        .addcol::<Float, _>((0..size).map(|v| v as f64 + 0.5).map(Some))?
        // .addcol::<Bool, _>((0..size).map(|v| v % 2 == 0))?;
        .addcol::<Strings, _>((0..size).map(|v| format!("number {}", v)).map(Some))?;

    // let tmp_dir = TempDir::new("framesexample")?;
    // let path = tmp_dir.path().join("test.csv");
    // println!("Writing CSV to {}", path.display());
    // let startw = time::Instant::now();
    // df1.write_csv(&path)?;
    // let tw = elapsed_secs(startw);
    // println!("Wrote in {}s ({:.0} lines/sec)", tw, size as f64 / tw);

    // println!("Reading CSV from {}", path.display());
    // let startr = time::Instant::now();
    // let df2: Frame3<Int, Float, Strings> = read_csv(&path).unwrap();
    // let tr = elapsed_secs(startr);
    // println!("Read in {}s ({:.0} lines/sec)", tr, size as f64 / tr);

    // println!("{}: {:#?}", Int::NAME, df2.get::<Int, _>().describe());

    let df1j = df1.clone();
    let startr = time::Instant::now();
    let df3 = df1j.inner_join::<Int, Int, _, _, _>(&df1);
    let tr = elapsed_secs(startr);
    println!("Joined to self in {}s", tr);

    let df1j = df1.clone();
    let startr = time::Instant::now();
    let _df4 = df1j.inner_join::<Int, Int, _, _, _>(&df1);
    let tr = elapsed_secs(startr);
    println!("Joined to self again in {}s", tr);

    println!("{}: {:#?}", Int::NAME, df3.get::<Int, _>().describe());

    // println!("{}", df2);

    // for (name, col) in df.itercols() {
    //     if let Some(d) = col.describe() {
    //     }
    // }

    Ok(())
}

fn elapsed_secs(t: time::Instant) -> f64 {
    let dur = time::Instant::now() - t;
    dur.as_secs() as f64 + dur.subsec_millis() as f64 / 1000.
}
