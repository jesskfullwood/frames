#[macro_use]
extern crate frames;
extern crate tempdir;

use frames::frame::Frame4;
use frames::*;
use tempdir::TempDir;

use std::time;

define_col!(Int, usize, ints);
define_col!(Int2, i64, ints);
define_col!(Float, f64, floats);
define_col!(Bool, bool, bools);
define_col!(Strings, String, strings);

const SIZE: usize = 10_000_000;

fn main() -> Result<()> {
    let t = time::Instant::now();
    let df1 = Frame::with::<Int, _>((0..SIZE).map(Some))
        .addcol::<Float, _>((0..SIZE).map(|v| v as f64 + 0.5).map(Some))?
        .addcol::<Bool, _>((0..SIZE).map(|v| if v % 2 == 0 { Some(v % 4 == 0) } else { None }))?
        .addcol::<Strings, _>((0..SIZE).map(|v| format!("number {}", v)).map(Some))?;
    let t = elapsed_secs(t);
    println!(
        "Created frame with {} columns and {} rows in {}s",
        df1.num_cols(),
        df1.len(),
        t
    );

    let t = time::Instant::now();
    let mut vec = Vec::with_capacity(100_000);
    for x in 0..SIZE {
        let elem = (
            Some(x),
            Some(x as f64 + 0.5),
            if x % 2 == 0 { Some(x % 4 == 0) } else { None },
            Some(format!("number {}", x)),
        );
        vec.push(elem);
    }
    let t = elapsed_secs(t);
    println!("Created equivalent vec with {} rows in {}s", vec.len(), t);

    let t = time::Instant::now();
    let clones = vec![df1.clone(); 100_000];
    let t = elapsed_secs(t);
    println!("Created {} clones in {}s", clones.len(), t);

    let tmp_dir = TempDir::new("framesexample")?;
    let path = tmp_dir.path().join("test.csv");
    println!("Writing CSV to {}", path.display());
    let startw = time::Instant::now();
    df1.write_csv(&path)?;
    let tw = elapsed_secs(startw);
    println!("Wrote in {}s ({:.0} lines/sec)", tw, SIZE as f64 / tw);

    println!("Reading CSV from {}", path.display());
    let startr = time::Instant::now();
    let df2: Frame4<Int, Float, Bool, Strings> = read_csv(&path).unwrap();
    let tr = elapsed_secs(startr);
    println!("Read in {}s ({:.0} lines/sec)", tr, SIZE as f64 / tr);

    println!("{}: {:#?}", Int::NAME, df2.get::<Int, _>().describe());

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

    Ok(())
}

fn elapsed_secs(t: time::Instant) -> f64 {
    let dur = time::Instant::now() - t;
    dur.as_secs() as f64 + dur.subsec_millis() as f64 / 1000.
}
