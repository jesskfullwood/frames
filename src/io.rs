use *;

use std::path::Path;
use std::io::{Cursor, Read};
use std::fs::File;

enum ColType {
    String,
    Bool,
    Float,
    Int
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
    let reader = csv::Reader::from_reader(reader);
    let headers = reader.headers()?;
    let ncols = headers.len();
    let csviter = reader.records();
    let row1 = csviter.next().ok_or_else(|| format_err!("No data"))??;
    let coltypes: Vec<_> = row1.iter().map(sniff_type).collect();
}

fn sniff_type(item: &str) -> ColType {
    use self::ColType::*;
    if item.parse::<i64>().is_ok() {
        Int
    } else if item.parse::<f64>().is_ok() {
        Float
    } else if item.parse::<bool>().is_ok() {
        Bool
    } else {
        String
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic() {
        let data =
"this is,a csv,with 3 cols
1,here,true
2,are,false
3,four,true
4,rows,false";
        let df = read_string(data).unwrap();

    }
}
