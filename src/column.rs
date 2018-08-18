use bit_vec::BitVec;
use num::traits::AsPrimitive;
use num::{self, Bounded, Num};
use smallvec;

use std;
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Rem, Sub};
use std::sync::Arc;

use {id, Array, StdResult};

// TODO benchmark smallvec vs Vec
pub(crate) type IndexVec = smallvec::SmallVec<[usize; 2]>;
type IndexMap<T> = HashMap<T, IndexVec>;
type IndexKeys<'a, T> = std::collections::hash_map::Keys<'a, T, IndexVec>;

// ### NamedColumn def and impls ###

#[derive(Clone, Debug, PartialEq)]
pub struct NamedColumn<T: ColId>(Column<T::Output>);

impl<T: ColId> NamedColumn<T> {
    pub(crate) fn new(anon: Column<T::Output>) -> Self {
        NamedColumn(anon)
    }

    pub(crate) fn with(from: impl IntoIterator<Item = T::Output>) -> NamedColumn<T> {
        NamedColumn::new(Column::new_notnull(from))
    }

    // Just a convenience method
    pub fn name(&self) -> &'static str {
        T::NAME
    }
}

impl<F, Col: ColId> From<F> for NamedColumn<Col>
where
    F: Into<Column<Col::Output>>,
{
    fn from(into_anon: F) -> NamedColumn<Col> {
        NamedColumn::new(into_anon.into())
    }
}

pub trait ColId: Copy {
    const NAME: &'static str;
    type Output;

    fn name() -> &'static str {
        Self::NAME
    }
}

impl<T: ColId> Deref for NamedColumn<T> {
    type Target = Column<T::Output>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ColId> DerefMut for NamedColumn<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, O> PartialEq<O> for NamedColumn<T>
where
    T: ColId,
    T::Output: PartialEq,
    Column<T::Output>: PartialEq<O>,
{
    fn eq(&self, other: &O) -> bool {
        &self.0 == other
    }
}

// ### Column def and impls ###

pub struct Column<T>(Arc<ColumnInner<T>>);

impl<T> Clone for Column<T> {
    // We add a custom clone because derive adds a T: Clone bound
    fn clone(&self) -> Self {
        Column(self.0.clone())
    }
}

#[derive(Clone)]
struct ColumnInner<T> {
    data: Array<ManuallyDrop<T>>,
    null_count: usize,
    null_vec: BitVec,
    index: RefCell<Option<IndexMap<T>>>,
}

impl<T> Drop for ColumnInner<T> {
    fn drop(&mut self) {
        self.data
            .iter_mut()
            .zip(self.null_vec.iter())
            .for_each(|(val, notnull)| {
                if notnull {
                    unsafe { ManuallyDrop::drop(val) }
                }
                // else val is actually just zeroed memory so don't drop
            })
    }
}

// impl<T, I: IntoIterator<Item = T>> From<I> for Column<T> {
//     fn from(iter: I) -> Column<T> {
//         Column::new_notnull(iter)
//     }
// }

impl<T, I: IntoIterator<Item = Option<T>>> From<I> for Column<T> {
    fn from(iter: I) -> Column<T> {
        Column::new(iter)
    }
}

impl<T: PartialEq> PartialEq for Column<T> {
    // We don't care whether the index exists so need custom impl
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .zip(other.iter())
            .all(|(left, right)| match (left, right) {
                (None, None) => true,
                (Some(_), None) => false,
                (None, Some(_)) => false,
                (Some(v1), Some(v2)) => v1 == v2,
            })
    }
}

impl<A, T> PartialEq<A> for Column<T>
where
    A: AsRef<[T]>,
    T: PartialEq,
{
    fn eq(&self, other: &A) -> bool {
        let slice: &[T] = other.as_ref();
        if self.count_null() > 0 || self.len() != slice.len() {
            return false;
        }
        self.iter()
            .filter_map(id)
            .zip(slice.iter())
            .all(|(l, r)| l == r)
    }
}

impl<T: Debug> Debug for Column<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        // This is very inefficient but we don't care because it's only for debugging
        let vals: Vec<String> = self
            .iter()
            .map(|v| {
                v.map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| String::from("NA"))
            }).collect();
        write!(
            f,
            "Column {{ indexed: {}, nulls: {}, vals: {:?} }}",
            self.0.index.borrow().is_some(),
            self.0.null_count,
            vals
        )
    }
}

impl<T: Sized> Column<T> {
    pub fn new(data: impl IntoIterator<Item = Option<T>>) -> Column<T> {
        let mut null_vec = BitVec::new();
        let mut null_count = 0;
        let mut arr = Vec::new();
        for v in data {
            push_maybe_null(v, &mut arr, &mut null_vec, &mut null_count);
        }
        Column(Arc::new(ColumnInner {
            null_count,
            null_vec,
            data: Array::new(arr),
            index: RefCell::new(None),
        }))
    }

    pub fn new_notnull(data: impl IntoIterator<Item = T>) -> Column<T> {
        // ManuallyDrop is a zero-cost wrapper so this should be safe
        let data = Array(data.into_iter().collect());
        let data = unsafe { std::mem::transmute::<Array<T>, Array<ManuallyDrop<T>>>(data) };
        Column(Arc::new(ColumnInner {
            null_count: 0,
            null_vec: BitVec::from_elem(data.len(), true),
            data,
            index: RefCell::new(None),
        }))
    }

    pub fn len(&self) -> usize {
        self.0.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shorthand for clone
    pub fn c(&self) -> Self {
        (*self).clone()
    }

    pub fn count_null(&self) -> usize {
        self.0.null_count
    }

    pub fn count_notnull(&self) -> usize {
        self.len() - self.0.null_count
    }

    pub fn is_indexed(&self) -> bool {
        self.0.index.borrow().is_some()
    }

    /// Returns wrapped value, or None if null,
    /// wrapped in bounds-check
    pub fn get(&self, ix: usize) -> Option<Option<&T>> {
        match self.0.null_vec.get(ix) {
            None => None, // out of bounds
            Some(true) => Some(Some(&self.0.data[ix])),
            Some(false) => Some(None),
        }
    }

    /// Iterate over the values of the column
    pub fn iter(&self) -> impl Iterator<Item = Option<&T>> + '_ {
        self.0
            .null_vec
            .iter()
            .zip(self.0.data.iter())
            .map(|(isvalid, v)| if isvalid { Some(v.deref()) } else { None })
    }

    /// Iterate over the none-null values of the column
    pub fn iter_notnull(&self) -> impl Iterator<Item = &T> + '_ {
        self.0
            .null_vec
            .iter()
            .zip(self.0.data.iter())
            .filter_map(|(isvalid, v)| if isvalid { Some(v.deref()) } else { None })
    }

    /// Construct a new column by applying `func` to contents
    pub fn map<R>(&self, func: impl Fn(Option<&T>) -> Option<R>) -> Column<R> {
        Column::new(self.iter().map(|v| func(v)))
    }

    /// Construct a new column by applying `func` to non-null contents
    /// (effectively filtering out nulls)
    pub fn map_notnull<R>(&self, func: impl Fn(&T) -> R) -> Column<R> {
        Column::new(self.iter().map(|v| {
            match v {
                Some(v) => Some(func(v)), // v.map(test) doesn't work for some reason
                None => None,
            }
        }))
    }

    pub fn mask(&self, test: impl Fn(&T) -> bool) -> Mask {
        let mask = self.map_notnull(test);
        mask.into()
    }

    pub fn iterate_indices(
        &self,
        iter: impl Iterator<Item = usize>,
    ) -> impl Iterator<Item = Option<&T>> {
        let data = &self.0.data;
        let nulls = &self.0.null_vec;
        iter.map(move |ix| if nulls[ix] { Some(&*data[ix]) } else { None })
    }
}

#[inline]
fn push_maybe_null<T>(
    val: Option<T>,
    data: &mut Vec<ManuallyDrop<T>>,
    null_vec: &mut BitVec,
    null_count: &mut usize,
) {
    match val {
        Some(v) => {
            null_vec.push(true);
            data.push(ManuallyDrop::new(v));
        }
        None => {
            null_vec.push(false);
            *null_count += 1;
            // TODO this is UB when we try to DROP it, will possibly segfault
            // Just use a default bound instead?
            let scary: T = unsafe { ::std::mem::zeroed() };
            data.push(ManuallyDrop::new(scary))
        }
    }
}

impl<T: Clone> Column<T> {
    pub fn filter(&self, func: impl Fn(Option<&T>) -> bool) -> Self {
        Column::new(
            self.iter()
                .filter_map(|v| if func(v) { Some(v.cloned()) } else { None }),
        )
    }

    pub fn filter_notnull(&self, func: impl Fn(&T) -> bool) -> Self {
        Column::new_notnull(self.iter_notnull().filter_map(|v| {
            if func(v) {
                Some(v.clone())
            } else {
                None
            }
        }))
    }

    /// Create new Column taking values from provided slice of indices
    pub(crate) fn copy_locs(&self, locs: &[usize]) -> Column<T> {
        Column::new(locs.iter().map(|&ix| self.get(ix).unwrap().cloned()))
    }

    /// Create new NamedColumn taking values from provided slice of indices,
    /// possibly interjecting nulls
    /// This function is mainly useful for joins
    pub(crate) fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Column<T> {
        Column::new(locs.iter().map(|&ix| {
            if let Some(ix) = ix {
                self.get(ix).unwrap().cloned()
            } else {
                None
            }
        }))
    }

    // TODO This basically exists to help with doing group-bys
    // might be a way to do things faster/more cleanly
    // It is guaranteed that each Vec<usize> is nonempty
    pub(crate) fn copy_first_locs(&self, locs: &[IndexVec]) -> Column<T> {
        let data: Vec<_> = locs
            .iter()
            .map(|inner| {
                let first = *inner.first().unwrap();
                // TODO We assume index is in bounds and value is not null
                self.get(first).unwrap().unwrap().clone()
            }).collect();
        Column::new_notnull(data)
    }

    // Filter collection from mask. Nulls are considered equivalent to false
    pub fn filter_mask(&self, mask: &Mask) -> Self {
        assert_eq!(self.len(), mask.len());
        Column::new(self.iter().zip(mask.iter()).filter_map(|(v, b)| {
            if b.unwrap_or(false) {
                Some(v.cloned())
            } else {
                None
            }
        }))
    }
}

impl<T: Hash + Clone + Eq> Column<T> {
    // TODO this seems to be very slow??
    pub fn build_index(&self) {
        if self.is_indexed() {
            return;
        }
        let mut index = IndexMap::with_capacity(self.len());
        for (ix, d) in self
            .iter()
            .enumerate()
            .filter_map(|(ix, d)| d.map(|d| (ix, d)))
        {
            let entry = index.entry(d.clone()).or_insert_with(IndexVec::default);
            entry.push(ix)
        }
        index.shrink_to_fit(); // we aren't touching this again
        *self.0.index.borrow_mut() = Some(index);
    }

    pub fn uniques(&self) -> UniqueIter<T> {
        self.build_index();
        UniqueIter {
            r: Ref::map(self.0.index.borrow(), |o| o.as_ref().unwrap()),
        }
    }

    pub(crate) fn inner_join_locs(&self, other: &Column<T>) -> (Vec<usize>, Vec<usize>) {
        other.build_index();
        let rborrow = other.0.index.borrow();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());
        self.iter()
            .enumerate()
            .filter_map(|(ix, lval)| lval.map(|d| (ix, d)))
            .for_each(|(lix, lval)| {
                if let Some(rixs) = rindex.get(lval) {
                    // We have found a join
                    rixs.iter().for_each(|&rix| {
                        leftout.push(lix);
                        rightout.push(rix);
                    })
                }
            });
        assert_eq!(leftout.len(), rightout.len());
        (leftout, rightout)
    }

    pub(crate) fn left_join_locs(&self, other: &Column<T>) -> (Vec<usize>, Vec<Option<usize>>) {
        other.build_index();
        let rborrow = other.0.index.borrow();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());

        for (lix, lvalo) in self.iter().enumerate() {
            if let Some(lval) = lvalo {
                if let Some(rixs) = rindex.get(lval) {
                    // we have a join
                    rixs.iter().for_each(|&rix| {
                        leftout.push(lix);
                        rightout.push(Some(rix));
                    })
                } else {
                    // we have no join
                    leftout.push(lix);
                    rightout.push(None);
                }
            } else {
                // we have a null
                leftout.push(lix);
                rightout.push(None);
            }
        }
        assert_eq!(leftout.len(), rightout.len());
        (leftout, rightout)
    }

    pub(crate) fn outer_join_locs(
        &self,
        other: &Column<T>,
    ) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        self.build_index();
        other.build_index();
        let lborrow = self.0.index.borrow();
        let lindex = lborrow.as_ref().unwrap();
        let rborrow = other.0.index.borrow();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());

        for (lix, lvalo) in self.iter().enumerate() {
            match lvalo {
                None => {
                    // Left value is null, so no joins
                    leftout.push(Some(lix));
                    rightout.push(None);
                }
                Some(lval) => {
                    let lixs = lindex.get(lval).unwrap();
                    match rindex.get(lval) {
                        None => {
                            // No join
                            leftout.push(Some(lix));
                            rightout.push(None);
                        }
                        Some(rixs) => {
                            // we have a join. Push each permutation of indexes
                            lixs.iter().for_each(|&lix| {
                                rixs.iter().for_each(|&rix| {
                                    leftout.push(Some(lix));
                                    rightout.push(Some(rix));
                                })
                            })
                        }
                    }
                }
            }
        }
        for (rix, rvalo) in other.iter().enumerate() {
            match rvalo {
                None => {
                    // Right value is null, add to output
                    leftout.push(None);
                    rightout.push(Some(rix));
                }
                Some(rval) => {
                    match lindex.get(rval) {
                        None => {
                            // No join, add in right index
                            leftout.push(None);
                            rightout.push(Some(rix));
                        }
                        Some(_) => {
                            // we have a join, but have already dealt with it above
                            // so there is nothing to be done the second time round
                        }
                    }
                }
            }
        }
        // Finally add in all the nulls from 'other', since they have been missed
        assert_eq!(leftout.len(), rightout.len());
        (leftout, rightout)
    }

    // TODO this seems very inefficient, what is it for?
    pub(crate) fn index_values(&self) -> Vec<IndexVec> {
        self.build_index();
        let borrow = self.0.index.borrow();
        let colix = borrow.as_ref().unwrap();
        colix.values().cloned().collect()
    }
}

// ### Primitive and columnwise op impls ###

macro_rules! impl_primitive_op {
    ($typ:ident, $func:ident) => {
        impl<T> $typ<T> for Column<T>
        where
            T: $typ + Clone,
        {
            type Output = Column<T::Output>;
            fn $func(self, rhs: T) -> Self::Output {
                //TODO map in-place?
                self.map_notnull(|v| T::$func(v.clone(), rhs.clone()))
            }
        }
    };
}

macro_rules! impl_columnwise_op {
    ($typ:ident, $func:ident) => {
        impl<T> $typ for Column<T>
        where
            T: $typ + Clone,
        {
            type Output = Column<T::Output>;
            fn $func(self, rhs: Column<T>) -> Self::Output {
                //TODO map in-place?
                if self.len() != rhs.len() {
                    panic!(
                        "Column lengths do not match ({} != {})",
                        self.len(),
                        rhs.len()
                    )
                }
                let iter = self
                    .iter()
                    .zip(rhs.iter())
                    .map(|(ov1, ov2)| match (ov1, ov2) {
                        (Some(v1), Some(v2)) => Some(T::$func(v1.clone(), v2.clone())),
                        _ => None,
                    });
                Column::new(iter)
            }
        }
    };
}

impl_primitive_op!(Add, add);
impl_primitive_op!(Sub, sub);
impl_primitive_op!(Mul, mul);
impl_primitive_op!(Div, div);
impl_primitive_op!(Rem, rem);

impl_columnwise_op!(Add, add);
impl_columnwise_op!(Sub, sub);
impl_columnwise_op!(Mul, mul);
impl_columnwise_op!(Div, div);
impl_columnwise_op!(Rem, rem);

impl<T> Neg for Column<T>
where
    T: Neg + Clone,
{
    type Output = Column<T::Output>;
    fn neg(self) -> Self::Output {
        self.map_notnull(|v| T::neg(v.clone()))
    }
}

// ### basic stats impls ###

// TODO use SIMD!
impl<T: Num + Copy> Column<T> {
    // TODO big risk of overflow for ints
    // use some kind of bigint
    pub fn sum(&self) -> T {
        self.iter()
            .filter_map(id)
            .fold(num::zero(), |acc, &v| acc + v)
    }
}

impl<T: Num + Copy + AsPrimitive<f64>> Column<T> {
    /// Calculate the mean of the collection. Ignores null values
    pub fn mean(&self) -> f64 {
        let s: f64 = self.sum().as_();
        s / self.len() as f64
    }

    /// Calculate the variance of the collection. Ignores null values
    pub fn variance(&self) -> f64 {
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;
        self.iter().filter_map(id).for_each(|n| {
            let n: f64 = n.as_();
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.count_notnull() as f64;
        sigmafxsqr / self.count_notnull() as f64 - mean * mean
    }

    /// Calculate the standard deviation of the collection. Ignores null values
    pub fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Calculate summary statistics for the column
    pub fn describe(&self) -> Describe<T>
    where
        T: Bounded + PartialOrd,
    {
        let mut min = T::max_value();
        let mut max = T::min_value();
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;
        let len = self.count_notnull();

        self.iter().filter_map(id).for_each(|&n| {
            if n < min {
                min = n;
            }
            if n > max {
                max = n
            }
            let n = n.as_();
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / len as f64;
        let variance = sigmafxsqr / len as f64 - mean * mean;
        Describe {
            len: self.len(),
            null_count: self.count_null(),
            min,
            max,
            mean,
            stdev: variance.sqrt(),
        }
    }
}

// TODO document usage
#[macro_export]
macro_rules! col {
    ($($vals:tt),*) => {
        {
            let mut v = Vec::new();
            $(
                let val = wrap_val!($vals);
                v.push(val);
            )*
            Column::from(v)
        }
    }
}

#[allow(unused_macros)]
macro_rules! wrap_val {
    (None) => {
        None
    };
    ($val:tt) => {
        Some($val)
    };
}

pub struct Mask {
    mask: Column<bool>,
    true_count: usize,
}

// TODO mask::from(vec<Bool>)
impl Mask {
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = Option<bool>> + '_ {
        self.mask.iter().map(|b| b.cloned())
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }
}

impl From<Column<bool>> for Mask {
    fn from(mask: Column<bool>) -> Self {
        let true_count = mask
            .iter()
            .fold(0, |acc, v| if *v.unwrap_or(&false) { acc + 1 } else { acc });
        Mask { mask, true_count }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Describe<N: Num> {
    pub len: usize,
    pub null_count: usize,
    pub min: N,
    // TODO do we want these?
    // pub first_quartile: N,
    // pub median: N,
    // pub third_quartile: N,
    // pub mode: (N, usize),
    pub max: N,
    pub mean: f64,
    pub stdev: f64,
}

// TODO should probably adapt this to just ho
pub struct UniqueIter<'a, T: 'a>
where
    T: Eq + Hash,
{
    r: Ref<'a, IndexMap<T>>,
}

impl<'a, 'b: 'a, T: 'a> IntoIterator for &'b UniqueIter<'a, T>
where
    T: Eq + Hash,
{
    type IntoIter = IndexKeys<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> IndexKeys<'a, T> {
        self.r.keys()
    }
}

// TODO this only works for single idents, ie "my string column" is not allowed
#[macro_export]
macro_rules! define_col {
    ($tyname:ident, $typ:ty) => {
        define_col!($tyname, $typ, $tyname);
    };
    ($tyname:ident, $typ:ty, $name:ident) => {
        // This type is just a marker and cannot be instantiated
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $tyname;
        impl ColId for $tyname {
            const NAME: &'static str = stringify!($name);
            type Output = $typ;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_with_nulls() {
        let c = Column::from((0..5).map(|v| if v % 2 == 0 { Some(v) } else { None }));
        assert_eq!(c.len(), 5);
        assert_eq!(c.count_null(), 2);
        let vals: Vec<Option<u32>> = c.iter().map(|v| v.cloned()).collect();
        assert_eq!(vals, vec![Some(0), None, Some(2), None, Some(4)]);

        let c2 = c.map_notnull(|v| *v);
        assert_eq!(c, c2);

        let c3 = c.map(|v| v.map(|u| u * u));
        let vals: Vec<Option<u32>> = c3.iter().map(|v| v.cloned()).collect();
        assert_eq!(vals, vec![Some(0), None, Some(4), None, Some(16)]);
    }

    #[test]
    fn test_build_null_strings() {
        let words = vec![
            Some(String::from("hi")),
            None,
            Some(String::from("there")),
            None,
            None,
        ];
        let c = Column::new(words.into_iter());
        assert_eq!(c.len(), 5);
        assert_eq!(c.count_null(), 3);
        assert_eq!(c.count_notnull(), 2);
        drop(c)
    }

    #[test]
    fn test_safely_drop() {
        use std::rc::Rc;
        use std::sync::Arc;
        // contains no nulls
        let c1 = Column::new_notnull(vec![Arc::new(10), Arc::new(20)]);
        drop(c1);
        // contains nulls -> segfaults!
        let c2 = Column::from(vec![Some(Rc::new(1)), None]);
        drop(c2);
    }

    #[test]
    fn test_col_macro() {
        let c = col![1, 2, 3, None, 4];
        assert_eq!(
            Column::from(vec![Some(1), Some(2), Some(3), None, Some(4)]),
            c,
        );
    }

    #[test]
    fn test_unique_iter() {
        let c = col![1, 2, 3, None, 4, 3, 4, 1, None, 5, 2];
        let mut keys: Vec<_> = c.uniques().into_iter().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_namedcol_name() {
        define_col!(Me, i64, mine);
        let c: NamedColumn<Me> = col![].into();
        assert_eq!(Me::name(), "mine");
        assert_eq!(c.name(), "mine");
    }

    #[test]
    fn test_basic_ops() {
        let c1 = col![1, 2, 3, None, 4, 3, 4, 1, None, 5, 2];
        let c2 = (c1.c() + 2) * 2;
        let c3 = (c2 / 2) - 2;
        assert_eq!(c1, c3);
        let c4 = c3.c() * c3 - c1;
        assert_eq!(c4, col![0, 2, 6, None, 12, 6, 12, 0, None, 20, 2]);
    }
}
