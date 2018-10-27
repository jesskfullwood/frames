use bit_vec::BitVec;
use num::traits::AsPrimitive;
use num::{self, Bounded, Num};
use smallvec;

use std;
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};
use std::mem::MaybeUninit;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Rem, Sub};
use std::sync::{RwLock, RwLockReadGuard};

use {id, StdResult};

// TODO benchmark smallvec vs Vec
pub(crate) type IndexVec = smallvec::SmallVec<[usize; 2]>;
type IndexMap<T> = BTreeMap<T, IndexVec>;
type IndexKeys<'a, T> = std::collections::btree_map::Keys<'a, T, IndexVec>;

/// A trait which tags a column with a name and type
pub trait ColId: Copy {
    const NAME: &'static str;
    type Output;

    fn name() -> &'static str {
        Self::NAME
    }
}

// ### NamedColumn def and impls ###

#[derive(Clone, Debug, PartialEq)]
pub struct NamedColumn<T: ColId>(Column<T::Output>);

impl<T: ColId> NamedColumn<T> {
    pub(crate) fn empty() -> Self {
        NamedColumn(Column::empty())
    }

    pub(crate) fn new(unnamed: Column<T::Output>) -> Self {
        NamedColumn(unnamed)
    }

    pub(crate) fn with(from: impl IntoIterator<Item = T::Output>) -> NamedColumn<T> {
        NamedColumn::new(Column::new(from))
    }

    /// Get the column name
    pub fn name(&self) -> &'static str {
        T::NAME
    }

    /// Unwrap the underlying column
    pub fn into_inner(self) -> Column<T::Output> {
        self.0
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

pub struct Column<T> {
    data: Vec<MaybeUninit<T>>,
    null_count: usize,
    valid_slots: BitVec,
    index: RwLock<Option<IndexMap<T>>>,
}

impl<T: Clone> Clone for Column<T> {
    // TODO this could be optimized by constructing directly
    // rather than going through an iterator
    fn clone(&self) -> Self {
        Column::new_null(self.iter_null().map(|maybe_val| maybe_val.cloned()))
    }
}

impl<T> Drop for Column<T> {
    fn drop(&mut self) {
        self.data
            .iter_mut()
            .zip(self.valid_slots.iter())
            .for_each(|(val, notnull)| {
                if notnull {
                    {
                        drop(unsafe { val.get_mut() })
                    }
                }
                // else val is actually just zeroed memory so don't drop
            })
    }
}

// pub trait IntoIterator where
//     <Self::IntoIter as Iterator>::Item == Self::Item, {
//     type Item;
//     type IntoIter: Iterator;
//     fn into_iter(self) -> Self::IntoIter;
// }

// impl<T, I: IntoIterator<Item = T>> From<I> for Column<T> {
//     fn from(iter: I) -> Column<T> {
//         Column::new_notnull(iter)
//     }
// }

impl<T, I: IntoIterator<Item = Option<T>>> From<I> for Column<T> {
    fn from(iter: I) -> Column<T> {
        Column::new_null(iter)
    }
}

impl<T: PartialEq> PartialEq for Column<T> {
    // We don't care whether the index exists so need custom impl
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter_null()
            .zip(other.iter_null())
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
        if self.count_nulls() > 0 || self.len() != slice.len() {
            return false;
        }
        self.iter_null()
            .filter_map(id)
            .zip(slice.iter())
            .all(|(l, r)| l == r)
    }
}

impl<T: Debug> Debug for Column<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        // This is very inefficient but we don't care because it's only for debugging
        let vals: Vec<String> = self
            .iter_null()
            .map(|v| {
                v.map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| String::from("NA"))
            })
            .collect();
        let vals = vals.join(", ");
        write!(
            f,
            "Column {{ indexed: {}, nulls: {}, vals: {} }}",
            self.index.read().unwrap().is_some(),
            self.null_count,
            vals
        )
    }
}

impl<T: Sized> Column<T> {
    pub fn empty() -> Column<T> {
        Column {
            null_count: 0,
            valid_slots: BitVec::new(),
            data: Vec::new(),
            index: RwLock::new(None),
        }
    }

    pub fn new(data: impl IntoIterator<Item = T>) -> Column<T> {
        // MaybeUninit is a zero-cost wrapper so cast this should be safe
        let data: Vec<MaybeUninit<T>> = data
            .into_iter()
            .map(|val| {
                let mut uninit = MaybeUninit::uninitialized();
                uninit.set(val);
                uninit
            })
            .collect();
        Column {
            null_count: 0,
            valid_slots: BitVec::from_elem(data.len(), true),
            data,
            index: RwLock::new(None),
        }
    }

    pub fn new_null(data: impl IntoIterator<Item = Option<T>>) -> Column<T> {
        let mut col = Column::empty();
        for v in data {
            col.insert(v);
        }
        col
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count number of null values. This is an O(1) operation
    pub fn count_nulls(&self) -> usize {
        self.null_count
    }

    /// Count number of non-null values. This is an O(1) operation
    pub fn count(&self) -> usize {
        self.len() - self.null_count
    }

    pub fn is_indexed(&self) -> bool {
        self.index.read().unwrap().is_some()
    }

    /// Returns wrapped value, or None if null,
    /// wrapped in bounds-check
    pub fn get(&self, ix: usize) -> Option<Option<&T>> {
        match self.valid_slots.get(ix) {
            None => None, // out of bounds
            Some(true) => Some(Some(unsafe { self.data[ix].get_ref() })),
            Some(false) => Some(None),
        }
    }

    pub fn insert(&mut self, val_opt: Option<T>) {
        match val_opt {
            Some(v) => {
                self.valid_slots.push(true);
                let mut m = MaybeUninit::uninitialized();
                m.set(v);
                self.data.push(m);
            }
            None => {
                self.valid_slots.push(false);
                self.null_count += 1;
                self.data.push(MaybeUninit::uninitialized());
            }
        }
    }

    /// Iterate over the non-null values of the column
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.valid_slots
            .iter()
            .zip(self.data.iter())
            .filter_map(|(isvalid, v)| {
                if isvalid {
                    Some(unsafe { v.get_ref() })
                } else {
                    None
                }
            })
    }

    /// Mutably iterate over the non-null values of the column
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.valid_slots
            .iter()
            .zip(self.data.iter_mut())
            .filter_map(|(isvalid, v)| {
                if isvalid {
                    Some(unsafe { v.get_mut() })
                } else {
                    None
                }
            })
    }

    /// Iterate over the values of the column, including nulls
    pub fn iter_null(&self) -> impl Iterator<Item = Option<&T>> + '_ {
        self.valid_slots
            .iter()
            .zip(self.data.iter())
            .map(|(isvalid, v)| {
                if isvalid {
                    Some(unsafe { v.get_ref() })
                } else {
                    None
                }
            })
    }

    /// Mutably iterate over the values of the column, including nulls
    pub fn iter_mut_null(&mut self) -> impl Iterator<Item = Option<&mut T>> + '_ {
        self.valid_slots
            .iter()
            .zip(self.data.iter_mut())
            .map(|(isvalid, v)| {
                if isvalid {
                    Some(unsafe { v.get_mut() })
                } else {
                    None
                }
            })
    }

    /// Convenience method to construct a new Column from application of `func`
    /// - nulls are propagated
    pub fn map<R>(&self, func: impl Fn(&T) -> R) -> Column<R> {
        Column::new_null(self.iter_null().map(|v| {
            match v {
                Some(v) => Some(func(v)), // v.map(test) doesn't work for some reason
                None => None,
            }
        }))
    }

    /// Convenience method to construct a new Column from application of `func`
    pub fn map_null<R>(&self, func: impl Fn(Option<&T>) -> Option<R>) -> Column<R> {
        Column::new_null(self.iter_null().map(|v| func(v)))
    }

    /// Apply `func` to contents in-place.
    /// This is the only way to turn nulls into values (and vice-versa) in-place
    pub fn map_in_place<R>(&mut self, func: impl Fn(Option<T>) -> Option<T>) {
        unimplemented!()
    }

    pub fn mask(&self, test: impl Fn(&T) -> bool) -> Mask {
        let mask = self.map(test);
        mask.into()
    }

    pub fn iterate_indices(
        &self,
        iter: impl Iterator<Item = usize>,
    ) -> impl Iterator<Item = Option<&T>> {
        let data = &self.data;
        let valid_slots = &self.valid_slots;
        iter.map(move |ix| {
            if valid_slots[ix] {
                Some(unsafe { data[ix].get_ref() })
            } else {
                None
            }
        })
    }
}

impl<T: Clone> Column<T> {
    /// Filter values with supplied function, creating a new Column.
    /// Nulls are left unchanged
    pub fn filter(&self, func: impl Fn(&T) -> bool) -> Self {
        Column::new_null(self.iter_null().filter_map(|v| match v {
            Some(v) => {
                if func(v) {
                    Some(Some(v.clone()))
                } else {
                    None
                }
            }
            None => Some(None),
        }))
    }

    /// Create new Column taking values from provided slice of indices
    pub(crate) fn copy_locs(&self, locs: &[usize]) -> Column<T> {
        Column::new_null(locs.iter().map(|&ix| self.get(ix).unwrap().cloned()))
    }

    /// Create new NamedColumn taking values from provided slice of indices,
    /// possibly interjecting nulls
    /// This function is mainly useful for joins
    pub(crate) fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Column<T> {
        Column::new_null(locs.iter().map(|&ix| {
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
            })
            .collect();
        Column::new(data)
    }

    // Filter collection from mask. Nulls are considered equivalent to false
    pub fn filter_mask(&self, mask: &Mask) -> Self {
        assert_eq!(self.len(), mask.len());
        Column::new_null(self.iter_null().zip(mask.iter()).filter_map(|(v, b)| {
            if b.unwrap_or(false) {
                Some(v.cloned())
            } else {
                None
            }
        }))
    }
}

impl<T: Ord + Clone + Eq> Column<T> {
    // TODO this seems to be very slow??
    pub fn build_index(&self) {
        if self.is_indexed() {
            return;
        }
        let mut index = IndexMap::new();
        for (ix, d) in self
            .iter_null()
            .enumerate()
            .filter_map(|(ix, d)| d.map(|d| (ix, d)))
        {
            let entry = index.entry(d.clone()).or_insert_with(IndexVec::default);
            entry.push(ix)
        }
        *self.index.write().unwrap() = Some(index);
    }

    #[allow(dead_code)]
    pub(crate) fn drop_index(&mut self) {
        // This is just for use with the benchmarks
        *self.index.write().unwrap() = None
    }

    pub fn uniques(&self) -> UniqueIter<T> {
        self.build_index();
        UniqueIter {
            guard: self.index.read().unwrap(),
        }
    }

    pub(crate) fn inner_join_locs(&self, other: &Column<T>) -> (Vec<usize>, Vec<usize>) {
        other.build_index();
        let rborrow = other.index.read().unwrap();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());
        self.iter_null()
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
        let rborrow = other.index.read().unwrap();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());

        for (lix, lvalo) in self.iter_null().enumerate() {
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
        let lborrow = self.index.read().unwrap();
        let lindex = lborrow.as_ref().unwrap();
        let rborrow = other.index.read().unwrap();
        let rindex = rborrow.as_ref().unwrap();
        let mut leftout = Vec::with_capacity(self.len()); // guess a preallocation
        let mut rightout = Vec::with_capacity(self.len());

        for (lix, lvalo) in self.iter_null().enumerate() {
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
        for (rix, rvalo) in other.iter_null().enumerate() {
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
        let borrow = self.index.read().unwrap();
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
                self.map(|v| T::$func(v.clone(), rhs.clone()))
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
                let iter =
                    self.iter_null()
                        .zip(rhs.iter_null())
                        .map(|(ov1, ov2)| match (ov1, ov2) {
                            (Some(v1), Some(v2)) => Some(T::$func(v1.clone(), v2.clone())),
                            _ => None,
                        });
                Column::new_null(iter)
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
        self.map(|v| T::neg(v.clone()))
    }
}

// ### basic stats impls ###

// TODO use SIMD!
impl<T: Num + Copy> Column<T> {
    // TODO big risk of overflow for ints
    // use some kind of bigint
    pub fn sum(&self) -> T {
        self.iter().fold(num::zero(), |acc, &v| acc + v)
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
        self.iter().for_each(|n| {
            let n: f64 = n.as_();
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.count() as f64;
        sigmafxsqr / self.count() as f64 - mean * mean
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
        let len = self.count();

        self.iter().for_each(|&n| {
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
            null_count: self.count_nulls(),
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
    ($($vals:tt),* $(,)*) => {
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
        self.mask.iter_null().map(|b| b.cloned())
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }
}

impl From<Column<bool>> for Mask {
    fn from(mask: Column<bool>) -> Self {
        let true_count = mask
            .iter_null()
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
    T: Eq + Ord,
{
    // Unfortunately this must be an `Option` because RwLockReadGuard
    // lacks a `map` method (unlike RefCell)
    guard: RwLockReadGuard<'a, Option<IndexMap<T>>>,
}

impl<'a, 'b: 'a, T: 'a> IntoIterator for &'b UniqueIter<'a, T>
where
    T: Eq + Ord,
{
    type IntoIter = IndexKeys<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> IndexKeys<'a, T> {
        // safe to unwrap the `Option<IndexMap>` as we know index has been built
        self.guard.as_ref().unwrap().keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_with_nulls() {
        let c = Column::from((0..5).map(|v| if v % 2 == 0 { Some(v) } else { None }));
        assert_eq!(c.len(), 5);
        assert_eq!(c.count_nulls(), 2);
        let vals: Vec<Option<u32>> = c.iter_null().map(|v| v.cloned()).collect();
        assert_eq!(vals, vec![Some(0), None, Some(2), None, Some(4)]);

        let c2 = c.map(|v| *v);
        assert_eq!(c, c2);

        let c3 = c.map_null(|v| v.map(|u| u * u));
        let vals: Vec<Option<u32>> = c3.iter_null().map(|v| v.cloned()).collect();
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
        let c = Column::new_null(words.into_iter());
        assert_eq!(c.len(), 5);
        assert_eq!(c.count_nulls(), 3);
        assert_eq!(c.count(), 2);
        drop(c)
    }

    #[test]
    fn test_safely_drop() {
        use std::rc::Rc;
        use std::sync::Arc;
        // contains no nulls
        let c1 = Column::new(vec![Arc::new(10), Arc::new(20)]);
        drop(c1);
        // contains nulls
        let c2 = Column::from(vec![Some(Rc::new(1)), None]);
        drop(c2);
        let c2 = Column::from(vec![Some(()), None]);
        drop(c2);
    }

    #[test]
    fn test_col_macro() {
        let c = col![1, 2, 3, NA, 4];
        assert_eq!(
            Column::from(vec![Some(1), Some(2), Some(3), None, Some(4)]),
            c,
        );
    }

    #[test]
    fn test_unique_iter() {
        let c = col![1, 2, 3, NA, 4, 3, 4, 1, NA, 5, 2];
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
        let c1 = col![1, 2, 3, NA, 4, 3, 4, 1, NA, 5, 2];
        let c2 = (c1.clone() + 2) * 2;
        let c3 = (c2 / 2) - 2;
        assert_eq!(c1, c3);
        let c4 = c3.clone() * c3 - c1;
        assert_eq!(c4, col![0, 2, 6, NA, 12, 6, 12, 0, NA, 20, 2]);
    }
}
