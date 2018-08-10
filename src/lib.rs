#![recursion_limit = "128"]

#[macro_use]
extern crate failure;
extern crate bit_vec;
extern crate csv;
extern crate num;
extern crate ordered_float;

use std::cell::RefCell;
use std::collections::{HashMap};
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Index;

use num::traits::AsPrimitive;
use num::Num;

use collection::Collection;

pub mod collection;
pub mod frame;
mod frame_alias;
pub mod io;

type StdResult<T, E> = std::result::Result<T, E>;
pub type Result<T> = StdResult<T, failure::Error>;
type IndexMap<T> = HashMap<T, Vec<usize>>;

// TODO Pretty-printing of Frame

// Helper function for filter_map (filters out nulls)
pub(crate) fn id<T>(v: T) -> T {
    v
}

/// NewType wrapper around Vec
#[derive(Debug, Clone)]
struct Array<T>(Vec<T>);

impl<T> Array<T> {
    #[inline]
    fn new(data: Vec<T>) -> Self {
        Array(data)
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    #[inline]
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Index<usize> for Array<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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

pub struct Mask {
    mask: Collection<bool>,
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
}

impl From<Collection<bool>> for Mask {
    fn from(mask: Collection<bool>) -> Self {
        let true_count = mask
            .iter()
            .fold(0, |acc, v| if *v.unwrap_or(&false) { acc + 1 } else { acc });
        Mask { mask, true_count }
    }
}
