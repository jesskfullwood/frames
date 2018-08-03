extern crate frames;
extern crate tempdir;

use frames::*;
use tempdir::TempDir;

use std::time;

fn main() -> Result<()> {
    let size = 10_000_000;
    let mut df = DataFrame::new();
    df.setcol("ints", Column::from((0..size).collect::<Vec<_>>()))?;
    df.setcol(
        "floats",
        Column::from((0..size).map(|v| v as f64 + 0.5).collect::<Vec<_>>()),
    )?;
    df.setcol(
        "bools",
        Column::from((0..size).map(|v| v % 2 == 0).collect::<Vec<_>>()),
    )?;
    df.setcol(
        "strings",
        Column::from(
            (0..size)
                .map(|v| format!("number {}", v))
                .collect::<Vec<_>>(),
        ),
    )?;

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
    let df2 = frames::io::read_csv(&path).unwrap();
    let durr = time::Instant::now() - startr;
    let tr = durr.as_secs() as f64 + durr.subsec_millis() as f64 / 1000.;
    println!("Read in {}s ({:.0} lines/sec)", tr, size as f64 / tr);

    println!("{}", df2);

    for (name, col) in df.itercols() {
        if let Some(d) = col.describe() {
            println!("{}: {:?}", name, d)
        }
    }

    Ok(())
}
