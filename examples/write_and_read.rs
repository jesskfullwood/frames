extern crate frames;
extern crate tempdir;

use frames::*;
use tempdir::TempDir;

fn main() -> Result<()> {
    let size = 10000;
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
    df.write_csv(&path)?;

    let df2 = frames::io::read_csv(&path).unwrap();

    println!("{}", df);
    println!("{}", df2);

    for (name, col) in df.itercols() {
        if let Some(d) = col.describe() {
            println!("{}: {:?}", name, d)
        }
    }

    Ok(())
}
