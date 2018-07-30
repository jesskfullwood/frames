#[macro_use]
extern crate failure;
extern crate num;
extern crate ordered_float;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Index;
use std::sync::Arc;

use num::traits::AsPrimitive;
use num::Num;
use ordered_float::OrderedFloat;

type StdResult<T, E> = std::result::Result<T, E>;
type Result<T> = StdResult<T, failure::Error>;
type IndexMap<T> = HashMap<T, Vec<usize>>;
type Array<T> = Vec<T>;

// TODO "TypedFrame" with a custom-derive?
// TODO Pretty-printing of DataFrame

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
            len: 0,
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

    pub fn addcol(&mut self, name: impl Into<String>, col: impl Into<Column>) -> Result<()> {
        let name = name.into();
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

    fn getcol(&self, key: &str) -> Result<&Column> {
        self.cols
            .get(key)
            .ok_or_else(|| format_err!("Column {} not found", key))
    }

    pub fn inner_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let col = self.getcol(on)?;
        let othercol = other.getcol(on)?;
        // get requisite left and right join indices
        let (leftix, rightix) = col.inner_join_locs(othercol);

        let mut newdf = DataFrame::new();

        // TODO actually we should add these in ORDER
        // make new collections from 'self' cols
        for (colname, col) in &self.cols {
            let newcol = col.copy_locs(&leftix);
            newdf.addcol(colname.clone(), newcol).unwrap();
        }
        // make new collections from 'other' cols
        for (colname, col) in &other.cols {
            if colname == on {
                // already done
                continue;
            }
            let newcol = col.copy_locs(&rightix);
            newdf.addcol(colname.clone(), newcol).unwrap();
        }
        Ok(newdf)
    }
}

impl<'a> Index<&'a str> for DataFrame {
    type Output = Column;

    fn index(&self, name: &'a str) -> &Column {
        &self.cols[name]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    inner: Arc<ColumnInner>,
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

    fn copy_locs(&self, locs: &[usize]) -> Self {
        Self {
            inner: Arc::new(self.inner.copy_locs(locs)),
        }
    }

    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ColumnInner {
    Bool(Collection<bool>),
    I32(Collection<i32>),
    String(Collection<String>),
    Float(Collection<OrderedFloat<f64>>),
}

// TODO can we make the function bit just an expression? (with args provided to expr if necessary?)
macro_rules! column_apply {
    ($fname:ident, $rtn:ty $(,$arg:ident: $argty:ty)* => $func:expr ) => {
        fn $fname(self: &Self, $($arg:$argty),*) -> $rtn {
            use ColumnInner::*;
            match self {
                  Bool(c) => $func(&c $(,$arg)*),
                   I32(c) => $func(&c $(,$arg)*),
                String(c) => $func(&c $(,$arg)*),
                 Float(c) => $func(&c $(,$arg)*),
            }
        }
    }
}

macro_rules! column_apply_pair {
    ($fname:ident, $rtn:ty $(,$arg:ident: $argty:ty)* => $func:expr ) => {
        fn $fname(self: &Self, other: &Self, $($arg:$argty),*) -> $rtn {
            use ColumnInner::*;
            match (self, other) {
                (Bool(c1), Bool(c2))     => $func(&c1, &c2 $(,$arg)*),
                (I32(c1), I32(c2))       => $func(&c1, &c2 $(,$arg)*),
                (String(c1), String(c2)) => $func(&c1, &c2 $(,$arg)*),
                (Float(c1), Float(c2))   => $func(&c1, &c2 $(,$arg)*),
                _ => panic!("Mismatching column types"),
            }
        }
    }
}

impl ColumnInner {
    column_apply!(len, usize => Collection::len);
    column_apply!(has_index, bool => Collection::has_index);
    column_apply!(build_index, () => Collection::build_index);
    column_apply_pair!(inner_join_locs, (Vec<usize>, Vec<usize>) => Collection::inner_join_locs);

    fn copy_locs(&self, locs: &[usize]) -> Self {
        use ColumnInner::*;
        match self {
            Bool(c) => Bool(c.copy_locs(locs)),
            I32(c) => I32(c.copy_locs(locs)),
            String(c) => String(c.copy_locs(locs)),
            Float(c) => Float(c.copy_locs(locs)),
        }
    }

    fn matching_types(&self, other: &ColumnInner) -> bool {
        match (self, other) {
            (ColumnInner::Bool(_), ColumnInner::Bool(_)) => true,
            (ColumnInner::I32(_), ColumnInner::I32(_)) => true,
            (ColumnInner::String(_), ColumnInner::String(_)) => true,
            _ => false,
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
    index: RefCell<Option<IndexMap<T>>>,
}

impl<T: PartialEq> PartialEq for Collection<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T> Debug for Collection<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        write!(f, "Collection")
    }
}

impl<T> Collection<T> {
    fn new(data: Array<T>) -> Collection<T> {
        Collection {
            data,
            index: RefCell::new(None),
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

impl<T: Clone> Collection<T> {
    /// Create new Collection taking values from provided slice of indices
    fn copy_locs(&self, locs: &[usize]) -> Collection<T> {
        let data = locs.iter().map(|&ix| self.data[ix].clone()).collect();
        Collection::new(data)
    }
}

impl<T: Hash + Clone + Eq> Collection<T> {
    fn build_index(&self) {
        if self.has_index() {
            return;
        }
        let mut index = IndexMap::new();
        for (ix, d) in self.data.iter().enumerate() {
            let entry = index.entry(d.clone()).or_insert(Vec::new());
            entry.push(ix)
        }
        *self.index.borrow_mut() = Some(index);
    }

    fn inner_join_locs(&self, other: &Collection<T>) -> (Vec<usize>, Vec<usize>) {
        // TODO if "other" is already indexed, we can skip this step
        self.build_index();

        let borrow = self.index.borrow();
        let colix = borrow.as_ref().unwrap();
        let mut pair: Vec<(usize, usize)> = Vec::new();
        for (rix, val) in other.iter().enumerate() {
            if let Some(lixs) = colix.get(val) {
                lixs.iter().for_each(|&lix| pair.push((lix, rix)))
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
}

impl<T: Num + Copy> Collection<T> {
    fn sum(&self) -> T {
        self.data.iter().fold(num::zero(), |acc, &v| acc + v)
    }
}

impl<T: Num + Copy + AsPrimitive<f64>> Collection<T> {
    fn mean(&self) -> f64 {
        let s: f64 = self.sum().as_();
        s / self.len() as f64
    }

    fn variance(&self) -> f64 {
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;
        self.data.iter().for_each(|n| {
            let n: f64 = n.as_();
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.len() as f64;
        sigmafxsqr / self.len() as f64 - mean * mean
    }

    fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }
}

macro_rules! impl_column_from {
    ($fromenum:ident, $fromty:ty) => {
        impl From<Array<$fromty>> for Column {
            fn from(arr: Array<$fromty>) -> Column {
                Column {
                    inner: Arc::new(ColumnInner::$fromenum(Collection::new(arr))),
                }
            }
        }
    };
}

impl_column_from!(I32, i32);
impl_column_from!(String, String);
impl_column_from!(Bool, bool);
// TODO this impl is no good, we want to pass simple f64s
impl_column_from!(Float, OrderedFloat<f64>);

impl From<Array<f64>> for Column {
    fn from(arr: Array<f64>) -> Column {
        let arr = unsafe { std::mem::transmute::<_, Array<OrderedFloat<f64>>>(arr) };
        Column {
            inner: Arc::new(ColumnInner::Float(Collection::new(arr))),
        }
    }
}

#[test]
fn test_basic_features() {
    let mut df = DataFrame::new();
    df.addcol("c1", vec![1, 2, 3, 4, 5]).unwrap();
    df.addcol("c2", vec![2., 3., 4., 5., 6.]).unwrap();
    df.addcol("c3", vec![true, true, false, true, false])
        .unwrap();
    let words: Vec<_> = "a b c d e".split(' ').map(String::from).collect();
    df.addcol("c4", words).unwrap();

    // test wrong length
    let too_long = vec![1, 2, 3, 4, 5, 6];
    df.addcol("cX", too_long).unwrap_err();
    df.getcol("cX").unwrap_err();

    assert_eq!(df.len(), 5);
    assert_eq!(df.num_cols(), 4);

    // index
    let c1 = &df["c1"];
    assert_eq!(c1.len(), 5);
}

#[test]
fn test_join() {
    let mut df = DataFrame::new();
    df.addcol("c1", vec![1, 2, 3, 4]).unwrap();
    df.addcol("c2", vec![true, false, true, false]).unwrap();

    let mut df2 = DataFrame::new();
    df2.addcol("c1", vec![2, 4, 3, 3, 2, 5]).unwrap();
    let dfjoin = df.inner_join(&df2, "c1").unwrap();

    let c1 = dfjoin.getcol("c1").unwrap();
    let e1 = Column::from(vec![2, 2, 3, 3, 4]);
    assert_eq!(c1, &e1);

    let c2 = dfjoin.getcol("c2").unwrap();
    let e2 = Column::from(vec![false, false, true, true, false]);
    assert_eq!(c2, &e2);

    df.addcol("c3", vec![4., 3., 2., 1.]).unwrap();
    df2.addcol("c3", vec![1., 0., 0., 0., 0., 0.]).unwrap();
    let dfjoin2 = df.inner_join(&df2, "c3").unwrap();
    assert_eq!(dfjoin2.len(), 1);
}
