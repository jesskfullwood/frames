#[macro_use]
extern crate failure;
extern crate csv;
extern crate num;
extern crate ordered_float;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Index;

use num::traits::AsPrimitive;
use num::Num;
use ordered_float::OrderedFloat;

pub mod io;

type StdResult<T, E> = std::result::Result<T, E>;
pub type Result<T> = StdResult<T, failure::Error>;
type IndexMap<T> = HashMap<T, Vec<usize>>;
type Array<T> = Vec<T>;
type Float = OrderedFloat<f64>;
type Int = i64;

// TODO "TypedFrame" with a custom-derive? Using an HList?
// TODO Pretty-printing of DataFrame

#[derive(Clone, Debug)]
pub struct DataFrame {
    cols: HashMap<String, Column>,
    order: Vec<String>,
    len: usize,
}

impl<'a> Index<&'a str> for DataFrame {
    type Output = Column;

    fn index(&self, name: &'a str) -> &Column {
        &self.cols[name]
    }
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

    // TODO maybe this shouldn't allocate?
    pub fn colnames(&self) -> Vec<&str> {
        self.order.iter().map(|s| s.as_str()).collect()
    }

    // TODO maybe this should allocate
    pub fn coltypes(&self) -> Vec<ColType> {
        self.order
            .iter()
            .map(|name| self.getcol(name).unwrap().coltype())
            .collect()
    }

    pub fn build_index(&self, key: &str) {
        let col = self.cols.get(key).unwrap();
        col.build_index();
    }

    pub fn setcol(&mut self, name: impl Into<String>, col: impl Into<Column>) -> Result<()> {
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

    pub fn getcol(&self, key: &str) -> Result<&Column> {
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
            newdf.setcol(colname.clone(), newcol).unwrap();
        }
        // make new collections from 'other' cols
        for (colname, col) in &other.cols {
            if colname == on {
                // already done
                continue;
            }
            let newcol = col.copy_locs(&rightix);
            newdf.setcol(colname.clone(), newcol).unwrap();
        }
        Ok(newdf)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColType {
    Int,
    Float,
    Bool,
    String,
}

impl From<Collection<f64>> for Column {
    fn from(coll: Collection<f64>) -> Column {
        let coll = unsafe { std::mem::transmute::<_, Collection<Float>>(coll) };
        Column::Float(coll)
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    Bool(Collection<bool>),
    Int(Collection<Int>),
    String(Collection<String>),
    Float(Collection<Float>),
}

// TODO can we make the function bit just an expression? (with args provided to expr if necessary?)
macro_rules! column_apply {
    ($fname:ident, $rtn:ty $(,$arg:ident: $argty:ty)* => $func:expr ) => {
        fn $fname(self: &Self, $($arg:$argty),*) -> $rtn {
            use Column::*;
            match self {
                  Bool(c) => $func(&c $(,$arg)*),
                   Int(c) => $func(&c $(,$arg)*),
                String(c) => $func(&c $(,$arg)*),
                 Float(c) => $func(&c $(,$arg)*),
            }
        }
    }
}

macro_rules! column_apply_pair {
    ($fname:ident, $rtn:ty $(,$arg:ident: $argty:ty)* => $func:expr ) => {
        fn $fname(self: &Self, other: &Self, $($arg:$argty),*) -> $rtn {
            use Column::*;
            match (self, other) {
                (Bool(c1), Bool(c2))     => $func(&c1, &c2 $(,$arg)*),
                (Int(c1), Int(c2))       => $func(&c1, &c2 $(,$arg)*),
                (String(c1), String(c2)) => $func(&c1, &c2 $(,$arg)*),
                (Float(c1), Float(c2))   => $func(&c1, &c2 $(,$arg)*),
                _ => panic!("Mismatching column types"),
            }
        }
    }
}

impl Column {
    column_apply!(len, usize => Collection::len);
    column_apply!(has_index, bool => Collection::has_index);
    column_apply!(build_index, () => Collection::build_index);
    column_apply_pair!(inner_join_locs, (Vec<usize>, Vec<usize>) => Collection::inner_join_locs);

    fn coltype(&self) -> ColType {
        use ColType as CT;
        use Column::*;
        match self {
            Bool(_) => CT::Bool,
            Int(_) => CT::Int,
            String(_) => CT::String,
            Float(_) => CT::Float,
        }
    }

    fn copy_locs(&self, locs: &[usize]) -> Self {
        use Column::*;
        match self {
            Bool(c) => Bool(c.copy_locs(locs)),
            Int(c) => Int(c.copy_locs(locs)),
            String(c) => String(c.copy_locs(locs)),
            Float(c) => Float(c.copy_locs(locs)),
        }
    }

    fn matching_types(&self, other: &Column) -> bool {
        match (self, other) {
            (Column::Bool(_), Column::Bool(_)) => true,
            (Column::Int(_), Column::Int(_)) => true,
            (Column::String(_), Column::String(_)) => true,
            _ => false,
        }
    }

    fn apply_mask(&self, mask: &Mask) -> Self {
        use Column::*;
        if mask.len() != self.len() {
            panic!(
                "Mask length doesn't match ({}, expected {})",
                mask.len(),
                self.len()
            )
        }
        match self {
            // TODO be nice to call c.wrap()
            Bool(c) => Bool(c.apply_mask(mask)),
            Int(c) => Int(c.apply_mask(mask)),
            String(c) => String(c.apply_mask(mask)),
            Float(c) => Float(c.apply_mask(mask)),
        }
    }

    pub fn mask<I>(&self, test: impl Fn(&I) -> bool) -> Mask
    where
        Self: DynamicMap<I, bool>,
    {
        // Example use:
        // let mask = df["thing"].mask(|v| v > 10)
        // TODO Macro:
        // let mask = m!(df["thing"] > 10)
        Mask(self.map_typed(test))
    }

    pub fn map<T, R>(&self, f: impl Fn(&T) -> R) -> Column
    where
        Self: DynamicMap<T, R> + From<Collection<R>>,
    {
        Self::from(self.map_typed(f))
    }

    // TODO sum, filter, reduce

    pub fn describe(&self) -> Option<Describe> {
        match self {
            Column::Int(c) => Some(c.describe()),
            Column::Float(c) => Some(c.describe()),
        }
    }
}

#[doc(hidden)]
pub trait DynamicMap<T, R> {
    fn map_typed(&self, f: impl Fn(&T) -> R) -> Collection<R>;
}

macro_rules! dynamic_map_impl {
    ($raw:ty, $enum:ident) => {
        impl<R> DynamicMap<$raw, R> for Column {
            fn map_typed(&self, f: impl Fn(&$raw) -> R) -> Collection<R> {
                use Column::*;
                match self {
                    $enum(c) => c.map(f),
                    _ => panic!("Can't apply function to TODO column"),
                }
            }
        }
    };
}

dynamic_map_impl!(Int, Int);
dynamic_map_impl!(bool, Bool);
dynamic_map_impl!(String, String);

macro_rules! impl_column_from_array {
    ($fromenum:ident, $fromty:ty) => {
        impl From<Array<$fromty>> for Column {
            fn from(arr: Array<$fromty>) -> Column {
                Column::$fromenum(Collection::new(arr))
            }
        }
    };
}

impl_column_from_array!(Int, Int);
impl_column_from_array!(String, String);
impl_column_from_array!(Bool, bool);
impl_column_from_array!(Float, Float);

impl From<Array<f64>> for Column {
    fn from(arr: Array<f64>) -> Column {
        // This looks very dangerous, but, OrderedFloat is just a newtype with no extra
        // invariants, so it should be safe (prove me wrong?)
        let arr = unsafe { std::mem::transmute::<_, Array<Float>>(arr) };
        Column::Float(Collection::new(arr))
    }
}

macro_rules! impl_column_from_collection {
    ($fromenum:ident, $fromty:ty) => {
        impl From<Collection<$fromty>> for Column {
            fn from(coll: Collection<$fromty>) -> Column {
                Column::$fromenum(coll)
            }
        }
    };
}

impl_column_from_collection!(Int, Int);
impl_column_from_collection!(String, String);
impl_column_from_collection!(Bool, bool);
impl_column_from_collection!(Float, Float);

#[derive(Clone)]
pub struct Collection<T> {
    data: Array<T>,
    index: RefCell<Option<IndexMap<T>>>,
}

impl<T: PartialEq> PartialEq for Collection<T> {
    // We don't care if the indexes are the same
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Debug> Debug for Collection<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        write!(
            f,
            "Collection {{ indexed: {}, vals: {:?} }}",
            self.index.borrow().is_some(),
            self.data
        )
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

    fn map<R>(&self, test: impl Fn(&T) -> R) -> Collection<R> {
        Collection::new(self.iter().map(test).collect())
    }
}

impl<T: Clone> Collection<T> {
    /// Create new Collection taking values from provided slice of indices
    fn copy_locs(&self, locs: &[usize]) -> Collection<T> {
        let data = locs.iter().map(|&ix| self.data[ix].clone()).collect();
        Collection::new(data)
    }

    pub fn apply_mask(&self, mask: &Mask) -> Self {
        assert_eq!(self.len(), mask.len());
        Collection::new(
            self.iter()
                .zip(mask.iter())
                .filter_map(|(v, &b)| if b { Some(v.clone()) } else { None })
                .collect(),
        )
    }
}

impl<T: Hash + Clone + Eq> Collection<T> {
    pub fn build_index(&self) {
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
    // TODO big risk of overflow for ints
    // use some kind of bigint
    pub fn sum(&self) -> T {
        self.data.iter().fold(num::zero(), |acc, &v| acc + v)
    }
}

impl<T: Num + Copy + AsPrimitive<f64>> Collection<T> {
    pub fn mean(&self) -> f64 {
        let s: f64 = self.sum().as_();
        s / self.len() as f64
    }

    pub fn variance(&self) -> f64 {
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

    pub fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn describe(&self) -> Describe {
        let mut min = std::f64::MAX;
        let mut max = std::f64::MIN;
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;

        self.data.iter().for_each(|n| {
            let n: f64 = n.as_();
            if n < min {
                min = n;
            }
            if n > max {
                max = n
            }
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.len() as f64;
        let variance = sigmafxsqr / self.len() as f64 - mean * mean;
        Describe {
            min,
            max,
            mean,
            stdev: variance.sqrt()
        }
    }
}

impl Collection<Float> {
    fn cast(&self) -> &Collection<Float> {
    }
}

pub struct Describe {
    // TODO Quartiles?
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stdev: f64,
}

pub struct Mask(Collection<bool>);

// TODO mask::from(vec<bool>)
impl Mask {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &bool> {
        self.0.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_features() {
        let mut df = DataFrame::new();
        df.setcol("c1", vec![1, 2, 3, 4, 5]).unwrap();
        df.setcol("c2", vec![2., 3., 4., 5., 6.]).unwrap();
        let col3 = Column::from(Collection::new(vec![true, true, false, true, false]));
        df.setcol("c3", col3);
        let words: Vec<_> = "a b c d e".split(' ').map(String::from).collect();
        df.setcol("c4", words).unwrap();

        // test wrong length
        let too_long = vec![1, 2, 3, 4, 5, 6];
        df.setcol("cX", too_long).unwrap_err();
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
        df.setcol("c1", vec![1, 2, 3, 4]).unwrap();
        df.setcol("c2", vec![true, false, true, false]).unwrap();

        let mut df2 = DataFrame::new();
        df2.setcol("c1", vec![2, 4, 3, 3, 2, 5]).unwrap();
        let dfjoin = df.inner_join(&df2, "c1").unwrap();

        let c1 = dfjoin.getcol("c1").unwrap();
        let e1 = Column::from(vec![2, 2, 3, 3, 4]);
        assert_eq!(c1, &e1);

        let c2 = dfjoin.getcol("c2").unwrap();
        let e2 = Column::from(vec![false, false, true, true, false]);
        assert_eq!(c2, &e2);

        // Join on floats (is this wise?)
        df.setcol("c3", vec![4., 3., 2., 1.]).unwrap();
        df2.setcol("c3", vec![1., 0., 0., 0., 0., 0.]).unwrap();
        let dfjoin2 = df.inner_join(&df2, "c3").unwrap();
        assert_eq!(dfjoin2.len(), 1);
    }

    #[test]
    fn test_mask() {
        let mut df = DataFrame::new();
        df.setcol("c1", vec![1, 2, 3, 4]).unwrap();
        let mask = df["c1"].mask(|&v: &Int| v > 2);
        let cfilt = df["c1"].apply_mask(&mask);
        assert_eq!(cfilt, Column::from(vec![3, 4]))
    }

    #[test]
    fn test_map() {
        let col = Column::from(vec![1, 2, 3, 4]);
        let colsqr = col.map(|v: &Int| v * v);
        assert_eq!(colsqr, Column::from(vec![1, 4, 9, 16]));

        // TODO make this work
        // let col = Column::from(vec![1., 2., 3., 4.]);
        // let colsqr = col.map(|v: &f64| v * v);
        // assert_eq!(colsqr, Column::from(vec![1., 4., 9., 16.]));
    }
}
