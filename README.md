# Frames

A DataFrame library for Rust

## Introduction

This library aims to provide a `DataFrame`-esque interface to tabular data,
i.e. an interface to perform common manipulations such as sum, filter, group-by and join
to in-memory columnar data.

A typical session might look something like this:

```
extern crate frames;
use frames::*;

fn main() -> Result<()> {
    let df1 = DataFrame::make((
        ("name", vec!["alice", "bob", "claire"]),
        ("age", vec![22, 23, 24]),
        ("height", vec![165.2, 188.0, 170.]),
    ))?;
}
```

## Why would I want to use DataFrames in Rust?

TODO
