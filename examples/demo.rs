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
    // Create frame

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

    // Clone many

    let t = time::Instant::now();
    let clones: Vec<_> = (0..10_000_000).map(|_| df1.clone()).collect();
    let t = elapsed_secs(t);
    println!("Created {} clones in {}s", clones.len(), t);

    // Describe column

    let t = time::Instant::now();
    let mean = df1.get(Int).describe().mean;
    assert_eq!(mean, 4999999.5);
    println!("Described Int column in {}s", elapsed_secs(t));
    let mean = df1.get(Float).describe().mean;
    assert_eq!(mean, 5000000.);
    println!("Described Float column in {}s", elapsed_secs(t));

    // Iter rows

    let t = time::Instant::now();
    let sum = df1.iter_rows().fold(
        0.,
        |acc, (_, flt, _, _)| if let Some(v) = flt { acc + v } else { acc },
    );
    assert_eq!(sum, 50000000000000.0);
    println!("Iterated rows in in {}s", elapsed_secs(t));

    let t = time::Instant::now();
    let sum = vec.iter().fold(
        0.,
        |acc, (_, flt, _, _)| if let Some(v) = flt { acc + v } else { acc },
    );
    assert_eq!(sum, 50000000000000.0);
    println!("Iterated alt rows in in {}s", elapsed_secs(t));

    // Build index

    let t = time::Instant::now();
    df1.get(Int).build_index();
    println!("Indexed Int column in {}s", elapsed_secs(t));

    // Join to self

    let df1j = df1.clone();
    let t = time::Instant::now();
    let _df3 = df1j.inner_join(&df1, Int, Int);
    println!("Joined to self in {}s", elapsed_secs(t));

    // Write to CSV

    let tmp_dir = TempDir::new("framesexample")?;
    let path = tmp_dir.path().join("test.csv");
    let t = time::Instant::now();
    df1.write_csv(&path)?;
    let tw = elapsed_secs(t);
    println!(
        "Wrote CSV to {} in {}s ({:.0} lines/sec)",
        path.display(),
        tw,
        SIZE as f64 / tw
    );

    // Read from CSV

    println!("Reading CSV from {}", path.display());
    let t = time::Instant::now();
    let _df2: Frame4<Int, Float, Bool, Strings> = read_csv(&path).unwrap();
    let tr = elapsed_secs(t);
    println!(
        "Read CSV from {} in {}s ({:.0} lines/sec)",
        path.display(),
        tr,
        SIZE as f64 / tr
    );

    Ok(())
}

fn elapsed_secs(t: time::Instant) -> f64 {
    let dur = time::Instant::now() - t;
    dur.as_secs() as f64 + dur.subsec_millis() as f64 / 1000.
}
