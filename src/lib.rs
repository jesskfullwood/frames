#[macro_use]
extern crate failure;

use std::collections::HashMap ;
use std::sync::Arc;
use std::hash::Hash;
use std::cell::RefCell;
use std::ops::{Index};
use std::fmt::{Debug, Formatter};

type StdResult<T, E> = std::result::Result<T, E>;
type Result<T> = StdResult<T, failure::Error>;
type IndexMap<T> = HashMap<T, Vec<usize>>;
type Array<T> = Vec<T>;

#[derive(Clone, Debug)]
pub struct DataFrame {
    cols: HashMap<String, Column>,
    order: Vec<String>,
    len: usize,
}

impl DataFrame {
    pub fn new() -> DataFrame {
        DataFrame {
            cols: HashMap::new(),
            order: Vec::new(),
            len: 0
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_cols(&self) -> usize {
        self.cols.len()
    }

    pub fn colnames(&self) -> Vec<&str> {
        self.order.iter().map(|s| s.as_str()).collect()
    }

    pub fn build_index(&self, key: &str) {
        let col = self.cols.get(key).unwrap();
        col.build_index();
    }

    pub fn add_col(&mut self, name: String, col: impl Into<Column>) -> Result<()> {
        let col = col.into();
        if self.num_cols() == 0 {
            self.len = col.len();
        } else if col.len() != self.len {
            bail!("Column is wrong length")
        }
        if let None = self.cols.insert(name.clone(), col) {
            self.order.push(name);
        }
        Ok(())
    }

    fn get_col(&self, key: &str) -> Result<&Column> {
        self.cols.get(key).ok_or_else(||format_err!("Column {} not found", key))
    }

    pub fn inner_join(&self, other: DataFrame, on: &str) -> Result<DataFrame> {
        let col = self.get_col(on)?;
        let othercol = other.get_col(on)?;
        // get requisite left and right join indices
        let (leftix, rightix) = col.inner_join_locs(othercol);
        let newdflen = leftix.len();
        let mut newdf = DataFrame::new();

        for (colname, colvals) in &self.cols {
            // make new collections
            // let mut newcol = Vec::with_capacity(newdflen);
        }
        unimplemented!();
    }

}

impl<'a> Index<&'a str> for DataFrame {
    type Output = Column;

    fn index(&self, name: &'a str) -> &Column {
        &self.cols[name]
    }
}

#[derive(Clone, Debug)]
pub struct Column {
    inner: Arc<ColumnInner>
}

impl Column {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn has_index(&self) -> bool {
        self.inner.has_index()
    }

    pub fn build_index(&self) {
        self.inner.build_index()
    }

    fn inner_join_locs(&self, other: &Column) -> (Vec<usize>, Vec<usize>) {
        self.inner.inner_join_locs(&*other.inner)
    }

    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }
}

#[derive(Clone, Debug)]
enum ColumnInner {
    Bool(Collection<bool>),
    I32(Collection<i32>),
    String(Collection<String>)
}

impl ColumnInner {
    fn len(&self) -> usize {
        match &self {
            ColumnInner::Bool(c) => c.len(),
            ColumnInner::I32(c) => c.len(),
            ColumnInner::String(c) => c.len(),
        }
    }

    fn has_index(&self) -> bool {
        match &self {
            ColumnInner::Bool(c) => c.has_index(),
            ColumnInner::I32(c) => c.has_index(),
            ColumnInner::String(c) => c.has_index(),
        }
    }

    fn build_index(&self) {
        match &self {
            ColumnInner::Bool(c) => c.build_index(),
            ColumnInner::I32(c) => c.build_index(),
            ColumnInner::String(c) => c.build_index(),
        }
    }

    fn matching_type(&self, other: &ColumnInner) -> bool {
        match (&self, other) {
            (ColumnInner::Bool(_), ColumnInner::Bool(_)) => true,
            (ColumnInner::I32(_), ColumnInner::I32(_)) => true,
            (ColumnInner::String(_), ColumnInner::String(_)) => true,
            _ => false
        }
    }

    fn inner_join_locs(&self, other: &ColumnInner) -> (Vec<usize>, Vec<usize>) {
        match (&self, other) {
            (ColumnInner::Bool(c1), ColumnInner::Bool(c2)) => c1.inner_join_locs(c2),
            (ColumnInner::I32(c1), ColumnInner::I32(c2)) => c1.inner_join_locs(c2),
            (ColumnInner::String(c1), ColumnInner::String(c2)) => c1.inner_join_locs(c2),
            _ => panic!("Mismatching column types")
        }
    }

    fn sum(&self) -> f64 {
        // How should this be done? Always f64? Precision a problem?
        unimplemented!()
    }
}

#[derive(Clone)]
struct Collection<T> {
    data: Array<T>,
    index: RefCell<Option<IndexMap<T>>>
}

impl<T> Debug for Collection<T>{
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        write!(f, "Collection")
    }
}

impl<T> Collection<T> {
    fn new(data: Array<T>) -> Collection<T> {
        Collection {
            data,
            index: RefCell::new(None)
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn has_index(&self) -> bool {
        self.index.borrow().is_some()
    }

    fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.data.iter()
    }
}

impl<T: Hash + Clone + Eq> Collection<T> {
    fn build_index(&self) {
        if self.has_index() {
            return
        }
        let mut index = IndexMap::new();
        for (ix, d) in self.data.iter().enumerate() {
            let entry = index.entry(d.clone()).or_insert(Vec::new());
            entry.push(ix)
        }
        *self.index.borrow_mut() = Some(index);
    }

    fn inner_join_locs(&self, other: &Collection<T>) -> (Vec<usize>, Vec<usize>) {
        self.build_index();
        let borrow = self.index.borrow();
        let colix = borrow.as_ref().unwrap();
        let mut pair: Vec<(usize, usize)> = Vec::new();
        for (rix, val) in other.iter().enumerate() {
            if let Some(lixs) = colix.get(val) {
                lixs.iter().for_each(|&lix| {
                    pair.push((lix, rix))
                })
            }
        }
        pair.sort_unstable();
        let mut left = Vec::with_capacity(pair.len());
        let mut right = Vec::with_capacity(pair.len());
        pair.iter().for_each(|&(l, r)| {
            left.push(l);
            right.push(r);
        });
        (left, right)
    }

    fn copy_locs(&self, locs: &[usize]) -> Collection<T> {
        let data = locs.iter().map(|&ix| self.data[ix].clone()).collect();
        Collection::new(data)
    }
}

impl From<Array<i32>> for Column {
    fn from(arr: Array<i32>) -> Column {
        Column {
            inner: Arc::new(ColumnInner::I32(Collection::new(arr)))
        }
    }
}

impl From<Array<String>> for Column {
    fn from(arr: Array<String>) -> Column {
        Column {
            inner: Arc::new(ColumnInner::String(Collection::new(arr)))
        }
    }
}

impl From<Array<bool>> for Column {
    fn from(arr: Array<bool>) -> Column {
        Column {
            inner: Arc::new(ColumnInner::Bool(Collection::new(arr)))
        }
    }
}

#[test]
fn test_new() {
    let mut df = DataFrame::new();
    let col = vec![1,2,3,4,5];
    df.add_col("mycol".into(), col).unwrap();

    let col2 = vec![2,3,4,5,6];
    df.add_col("mycol2".into(), col2).unwrap();

    // wrong length
    let col3 = vec![1,2,3,4,5,6];
    df.add_col("mycol3".into(), col3).unwrap_err();
    df.get_col("mycol3").unwrap_err();

    let c = df["mycol"].clone();

    println!("{:?}", df);

}
