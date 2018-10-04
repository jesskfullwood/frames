# Frames

A DataFrame library for Rust

## Introduction

This library aims to provide a typesafe `DataFrame`-esque interface to tabular data,
i.e. an interface to perform common manipulations such as sum, map, filter, group-by and join
to in-memory columnar data.

A typical session might look something like this:

```rust
extern crate frames;
use frames::*;

const SIZE: usize = 10_000_000;

fn main() {

    // Create frame
    let df1 = Frame::with::<Int, _>((0..SIZE).map(Some))
        .add::<Float, _>((0..SIZE).map(|v| v as f64 + 0.5).map(Some))?
        .add::<Bool, _>((0..SIZE).map(|v| if v % 2 == 0 { Some(v % 4 == 0) } else { None }))?
        .add::<Strings, _>((0..SIZE).map(|v| format!("number {}", v)).map(Some))?;

    // Describe column
    println!("{:?}", df1.get(Int).describe());

    // Iter rows and accumulate floats
    let sum = df1.iter_rows().fold(
        0.,
        |acc, (_, flt, _, _)| if let Some(v) = flt { acc + v } else { acc },
    );

    // Join to self
    let df2 = df1.clone();
    let df3 = df2.inner_join(&df1, Int, Int);

    // Write to CSV

    df1.write_csv("mydata.csv")?;
    Ok(())
}

```

## Why would I want to use DataFrames in Rust?

TODO Why indeed!
