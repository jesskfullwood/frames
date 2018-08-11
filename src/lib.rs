#![recursion_limit = "128"]

#[macro_use]
extern crate failure;
extern crate bit_vec;
extern crate csv;
extern crate num;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate ordered_float;

use std::collections::HashMap;
use std::ops::Index;

pub use column::Column;
pub use frame::{ColId, Frame};
pub use io::read_csv;

pub mod column;
pub mod frame;
mod frame_alias;
pub(crate) mod hlist;
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
