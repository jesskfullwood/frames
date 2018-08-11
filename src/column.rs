use bit_vec::BitVec;
use num::traits::AsPrimitive;
use num::{self, Num};

use std;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;

use {id, Array, IndexMap, StdResult};

#[derive(Clone)]
pub struct Column<T>(ColumnInner<T>);

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

impl<T, I: IntoIterator<Item = T>> From<I> for Column<T> {
    fn from(iter: I) -> Column<T> {
        Column::new_notnull(iter)
    }
}

// impl<T, I: IntoIterator<Item=Option<T>>> From<I> for Column<T> {
//     fn from(iter: I) -> Column<T> {
//         Column::new(iter)
//     }
// }

impl<T: PartialEq> PartialEq for Column<T> {
    // We don't care if the indexes are the same
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

impl<T: Debug> Debug for Column<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        // This is very inefficient but we don't care because it's only for debugging
        let vals: Vec<String> = self
            .iter()
            .map(|v| {
                v.map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| String::from("NA"))
            }).collect();;
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
        Column(ColumnInner {
            null_count,
            null_vec,
            data: Array::new(arr),
            index: RefCell::new(None),
        })
    }

    pub fn new_notnull(data: impl IntoIterator<Item = T>) -> Column<T> {
        // ManuallyDrop is a zero-cost wrapper so this should be safe
        let data = Array(data.into_iter().collect());
        let data = unsafe { std::mem::transmute::<Array<T>, Array<ManuallyDrop<T>>>(data) };
        Column(ColumnInner {
            null_count: 0,
            null_vec: BitVec::from_elem(data.len(), true),
            data,
            index: RefCell::new(None),
        })
    }

    pub fn len(&self) -> usize {
        self.0.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn count_null(&self) -> usize {
        self.0.null_count
    }

    pub fn count_notnull(&self) -> usize {
        self.len() - self.0.null_count
    }

    pub fn has_index(&self) -> bool {
        self.0.index.borrow().is_some()
    }

    // This is unsafe at the moment because it will invalidate
    // the index, if it exsts
    pub(crate) unsafe fn push(&mut self, val: Option<T>) {
        push_maybe_null(
            val,
            &mut self.0.data.0,
            &mut self.0.null_vec,
            &mut self.0.null_count,
        )
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

    pub fn iter(&self) -> impl Iterator<Item = Option<&T>> + '_ {
        self.0
            .null_vec
            .iter()
            .zip(self.0.data.iter())
            .map(|(isvalid, v)| if isvalid { Some(v.deref()) } else { None })
    }

    pub fn iter_notnull(&self) -> impl Iterator<Item = &T> + '_ {
        self.0
            .null_vec
            .iter()
            .zip(self.0.data.iter())
            .filter_map(|(isvalid, v)| if isvalid { Some(v.deref()) } else { None })
    }

    // TODO this is underused
    pub fn map<R>(&self, func: impl Fn(Option<&T>) -> Option<R>) -> Column<R> {
        Column::new(self.iter().map(|v| func(v)))
    }

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
            // TODO this is UB when we try to DROP it, will probably segfault
            let scary: T = unsafe { ::std::mem::zeroed() };
            data.push(ManuallyDrop::new(scary))
        }
    }
}

// We don't care whether the index exists so need custom impl
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

impl<T: Clone> Column<T> {
    /// Create new Column taking values from provided slice of indices
    pub(crate) fn copy_locs(&self, locs: &[usize]) -> Column<T> {
        Column::new(locs.iter().map(|&ix| self.get(ix).unwrap().cloned()))
    }

    /// Create new Column taking values from provided slice of indices,
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
    pub(crate) fn copy_first_locs(&self, locs: &[Vec<usize>]) -> Column<T> {
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
    // TODO Question: Should to location of nulls be indexed?
    // I think not - we already have the bit-vec
    pub fn build_index(&self) {
        if self.has_index() {
            return;
        }
        let mut index = IndexMap::new();
        for (ix, d) in self
            .iter()
            .enumerate()
            .filter_map(|(ix, d)| d.map(|d| (ix, d)))
        {
            let entry = index.entry(d.clone()).or_insert_with(Vec::new);
            entry.push(ix)
        }
        *self.0.index.borrow_mut() = Some(index);
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
                            // we have a join, so there is nothing to be done
                            // the second time round
                        }
                    }
                }
            }
        }
        // Finally add in all the nulls from 'other', since they have been missed
        assert_eq!(leftout.len(), rightout.len());
        (leftout, rightout)
    }

    pub(crate) fn index_values(&self) -> Vec<Vec<usize>> {
        self.build_index();
        let borrow = self.0.index.borrow();
        let colix = borrow.as_ref().unwrap();
        colix.values().cloned().collect()
    }
}

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
    pub fn describe(&self) -> Describe {
        let mut min = std::f64::MAX;
        let mut max = std::f64::MIN;
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;
        let len = self.count_notnull();

        self.iter().filter_map(id).for_each(|n| {
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
pub struct Describe {
    // TODO Quartiles?
    pub len: usize,
    pub null_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stdev: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_with_nulls() {
        let c = Column::new((0..5).map(|v| if v % 2 == 0 { Some(v) } else { None }));
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
        let c2 = Column::new(vec![Some(Rc::new(1)), None]);
        drop(c2);
    }
}
