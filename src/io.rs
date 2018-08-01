use *;

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

enum CollectionBuilder {
    String(Vec<String>),
    Bool(Vec<bool>),
    Float(Vec<Float>),
    Int(Vec<Int>),
}

impl CollectionBuilder {
    fn push(&mut self, record: &str) {
        use self::CollectionBuilder as CB;
        match self {
            CB::Float(v) => v.push(record.parse::<f64>().unwrap().into()),
            CB::Int(v) => v.push(record.parse::<Int>().unwrap().into()),
            CB::Bool(v) => v.push(record.parse::<bool>().unwrap().into()),
            CB::String(v) => v.push(record.into()),
        }
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
}

fn read_csv(path: impl AsRef<Path>) -> Result<DataFrame> {
    let f = File::open(path)?;
    read_reader(f)
}

fn read_string(data: &str) -> Result<DataFrame> {
    let cur = Cursor::new(data);
    read_reader(cur)
}

fn read_reader<R: Read>(reader: R) -> Result<DataFrame> {
    let mut reader = csv::Reader::from_reader(reader);
    let headers = reader.headers()?.clone();
    let mut csviter = reader.records();
    let row1 = csviter.next().ok_or_else(|| format_err!("No data"))??;
    let mut columns: Vec<_> = row1.iter().map(sniff_coltype).collect();
    for row in csviter {
        for (elem, col) in row?.iter().zip(columns.iter_mut()) {
            col.push(elem)
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

fn sniff_coltype(item: &str) -> CollectionBuilder {
    if let Ok(v) = item.parse::<i64>() {
        CollectionBuilder::Int(vec![v])
    } else if let Ok(v) = item.parse::<f64>() {
        CollectionBuilder::Float(vec![v.into()])
    } else if let Ok(v) = item.parse::<bool>() {
        CollectionBuilder::Bool(vec![v])
    } else {
        CollectionBuilder::String(vec![item.into()])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic() {
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
}
