#[macro_use]
extern crate failure;
extern crate csv;
extern crate num;
extern crate ordered_float;

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Index;

use num::traits::AsPrimitive;
use num::Num;
use ordered_float::OrderedFloat;

use collection::Collection;

pub mod collection;
// pub mod dframe;
pub mod io;

type StdResult<T, E> = std::result::Result<T, E>;
pub type Result<T> = StdResult<T, failure::Error>;
type IndexMap<T> = HashMap<T, Vec<usize>>;
pub(crate) type Array<T> = Vec<T>;
pub(crate) type Float = f64;
pub(crate) type Bool = bool;
pub(crate) type OrdFloat = OrderedFloat<f64>;
pub(crate) type Int = i64;

// TODO "TypedFrame" with a custom-derive? Using an HList?
// TODO Pretty-printing of DataFrame

#[derive(Clone, Debug, PartialEq)]
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

    pub fn getcol(&self, key: &str) -> Result<&Column> {
        self.cols
            .get(key)
            .ok_or_else(|| format_err!("Column {} not found", key))
    }

    pub fn get_collection<T: 'static>(&self, key: &str) -> Result<&Collection<T>> {
        let col = self.getcol(key)?;
        col.get_collection()
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

    /// Add the columns of another dataframe to this frame
    pub fn combine(&mut self, other: &DataFrame) -> Result<()> {
        if self.len() > 0 && self.len() != other.len() {
            bail!(
                "Mismatched lengths (expected {}, got {})",
                self.len(),
                other.len()
            )
        }
        {
            let mut nameset: HashSet<&str> = self.colnames().iter().map(|s| *s).collect();
            for name in other.colnames() {
                if !nameset.insert(name) {
                    bail!("Duplicate column name: {}", name);
                }
            }
        }
        for (name, col) in other.itercols() {
            self.setcol(name, col.clone()).unwrap();
        }
        Ok(())
    }

    /// Add an index to the named column
    /// If the column name is not recognized
    pub fn build_index(&self, name: &str) -> Result<()> {
        let col = self.cols
            .get(name)
            .ok_or_else(|| format_err!("Column '{}' not found", name))?;
        col.build_index();
        Ok(())
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

    pub fn itercols(&self) -> impl Iterator<Item = (&str, &Column)> {
        ColIter {
            df: &self,
            cur_ix: 0,
        }
    }
}

impl Display for DataFrame {
    // TODO print some actual values
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        write!(
            f,
            "DataFrame - {} columns, {} rows\n",
            self.num_cols(),
            self.len()
        )?;
        for (name, c) in self.itercols() {
            write!(f, "    {:10}: {:?}\n", name, c.coltype())?;
        }
        Ok(())
    }
}

impl<S, C> From<(S, C)> for DataFrame
where
    S: Into<String>,
    C: Into<Column>,
{
    fn from((name, col): (S, C)) -> DataFrame {
        let mut df = DataFrame::new();
        df.setcol(name.into(), col.into()).unwrap();
        df
    }
}

pub trait ToFrame<T> {
    fn make(T) -> Result<DataFrame>;
}

macro_rules! impl_make_dataframe {
    ($t1:ident $(,$t:ident)*) => {
        make_dataframe_inner!($t1 $(,$t)*);
        impl_make_dataframe!($($t),*);
    };
    () => {}
}

macro_rules! make_dataframe_inner {
    ($($t:ident),+)=> {
        impl< $($t,)+ > ToFrame<($($t,)+)> for DataFrame
            where $($t: Into<DataFrame>),+ {
            #[allow(non_snake_case)]
            fn make(($($t,)+): ($($t,)+)) -> Result<DataFrame> {
                let mut base = DataFrame::new();
                $(
                    let df = $t.into();
                    base.combine(&df)?;
                )+
                Ok(base)
            }
        }
    }
}

impl_make_dataframe!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16);

struct ColIter<'a> {
    df: &'a DataFrame,
    cur_ix: usize,
}

impl<'a> Iterator for ColIter<'a> {
    type Item = (&'a str, &'a Column);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_ix < self.df.order.len() {
            let name = &self.df.order[self.cur_ix];
            let col = &self.df[name];
            self.cur_ix += 1;
            Some((name, col))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColType {
    Int,
    Float,
    Bool,
    String,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    Bool(Collection<Bool>),
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

impl Column {
    column_apply!(len, usize => Collection::len);
    column_apply!(has_index, Bool => Collection::has_index);
    // column_apply!(build_index, () => Collection::build_index);
    // column_apply_pair!(inner_join_locs, (Vec<usize>, Vec<usize>) => Collection::inner_join_locs);

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

    fn build_index(&self) {
        use Column::*;
        match self {
            Bool(c) => c.build_index(),
            Int(c) => c.build_index(),
            String(c) => c.build_index(),
            Float(c) => c.as_ordered().build_index(),
        }
    }

    fn inner_join_locs(&self, other: &Self) -> (Vec<usize>, Vec<usize>) {
        use Column::*;
        match (self, other) {
            (Bool(c1), Bool(c2)) => c1.inner_join_locs(c2),
            (Int(c1), Int(c2)) => c1.inner_join_locs(c2),
            (String(c1), String(c2)) => c1.inner_join_locs(c2),
            (Float(c1), Float(c2)) => c1.as_ordered().inner_join_locs(c2.as_ordered()),
            _ => panic!("Mismatching column types"),
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

    pub fn apply_mask(&self, mask: &Mask) -> Self {
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

    fn get_collection<T: 'static>(&self) -> Result<&Collection<T>> {
        use Column::*;
        match self {
            Bool(c) => (c as &Any).downcast_ref::<Collection<T>>().ok_or_else(|| format_err!("wrong type")),
            Int(c) => (c as &Any).downcast_ref::<Collection<T>>().ok_or_else(|| format_err!("wrong type")),
            Float(c) => (c as &Any).downcast_ref::<Collection<T>>().ok_or_else(|| format_err!("wrong type")),
            String(c) => (c as &Any).downcast_ref::<Collection<T>>().ok_or_else(|| format_err!("wrong type")),
        }
    }

    pub fn mask<I>(&self, test: impl Fn(&I) -> Bool) -> Mask
    where
        Self: DynamicMap<I, Bool>,
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
            _ => None,
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
dynamic_map_impl!(Float, Float);
dynamic_map_impl!(Bool, Bool);
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
impl_column_from_array!(Bool, Bool);
impl_column_from_array!(Float, Float);

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
impl_column_from_collection!(Bool, Bool);
impl_column_from_collection!(Float, Float);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Describe {
    // TODO Quartiles?
    pub len: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stdev: f64,
}

pub struct Mask(Collection<Bool>);

// TODO mask::from(vec<Bool>)
impl Mask {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Bool> {
        self.0.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_features() {
        let mut df = DataFrame::from(("c1", vec![1, 2, 3, 4, 5]));
        df.setcol("c1", vec![1, 2, 3, 4, 5]).unwrap();
        df.setcol("c2", vec![2., 3., 4., 5., 6.]).unwrap();
        let col3 = Column::from(Collection::new(vec![true, true, false, true, false]));
        df.setcol("c3", col3).unwrap();
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
    fn test_construct_from_vecs() {
        let mut df = DataFrame::new();
        let df1 =
            DataFrame::make((("c1", vec![1, 2, 3, 4]), ("c2", vec![1., 2., 3., 4.]))).unwrap();
        let words: Vec<String> = "this is some words".split(' ').map(String::from).collect();
        let df2 = DataFrame::make((
            ("c3", words),
            ("c4", vec![1., 2., 3., 4.]),
            ("c5", vec![true, false, false, true]),
        )).unwrap();
        df.combine(&df1).unwrap();
        df.combine(&df2).unwrap();
        assert_eq!(df.num_cols(), 5);
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

        let col = Column::from(vec![1., 2.5, 3., 4.]);
        let colsqr = col.map(|v: &f64| v * v);
        assert_eq!(colsqr, Column::from(vec![1., 6.25, 9., 16.]));
    }

    #[test]
    fn test_get_collection() {
        let df = DataFrame::make((
            ("c1", vec![1,2,3,4]),
            ("c2", vec![true, false, true, false])
        )).unwrap();
        assert!(df.get_collection::<Int>("c1").is_ok());
        assert!(df.get_collection::<Bool>("c1").is_err());
        assert!(df.get_collection::<Bool>("c2").is_ok());
    }
}
