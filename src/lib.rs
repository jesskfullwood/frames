#![recursion_limit = "128"]
#![cfg_attr(feature = "unstable", feature(test))]

#[cfg(all(feature = "unstable", test))]
extern crate rand;
#[cfg(all(feature = "unstable", test))]
extern crate test;

#[macro_use]
extern crate failure;
extern crate bit_vec;
extern crate csv;
extern crate frunk;
extern crate num;
extern crate ordered_float;
extern crate serde;
extern crate smallvec;

use std::ops::Index;

pub use column::{ColId, Column, NamedColumn};
pub use frame::Frame;
pub use io::read_csv;

#[macro_use]
pub mod column;
#[macro_use]
pub mod frame;
mod frame_typedef;
pub(crate) mod hlist;
pub mod io;

type StdResult<T, E> = std::result::Result<T, E>;
pub type Result<T> = StdResult<T, failure::Error>;

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

#[cfg(all(feature = "unstable", test))]
mod benchmarks {
    use super::*;
    use frame::test_fixtures::*;
    use frame::Frame3;
    use test::Bencher;

    #[bench]
    fn bench_create_frame(b: &mut Bencher) {
        b.iter(|| {
            let f: Frame3<IntT, FloatT, BoolT> = frame![
                col![1, 2, 3, 4],
                col![1., 2., 3., 4.],
                col![true, false, true, false],
            ];
            f
        })
    }

    const IXSIZE: i64 = 50_000;

    #[bench]
    fn bench_index_column_sequence(b: &mut Bencher) {
        let mut c = Column::new((0..IXSIZE).map(Some));
        b.iter(|| {
            c.build_index();
            c.drop_index();
        })
    }

    #[bench]
    fn bench_index_column_random_with_10_dupes(b: &mut Bencher) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut c = Column::new((0..IXSIZE).map(|_| rng.gen_range(0, IXSIZE / 10)).map(Some));
        b.iter(|| {
            c.build_index();
            c.drop_index();
        })
    }

    #[bench]
    fn bench_inner_join(b: &mut Bencher) {
        let f1 = Frame::with((0..IXSIZE).map(Some));
        f1.get(IntT).build_index();
        let f2 = f1.clone();
        b.iter(|| f1.clone().inner_join(&f2, IntT, IntT))
    }
}
