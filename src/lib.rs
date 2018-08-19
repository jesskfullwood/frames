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

pub use column::{ColId, Column, NamedColumn};
pub use frame::Frame;
pub use io::read_csv;

mod array;
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

#[cfg(all(feature = "unstable", test))]
mod benchmarks {
    use super::*;
    use frame::test_fixtures::*;
    use frame::Frame3;
    use test::Bencher;

    const SIZE: i64 = 50_000;

    #[bench]
    fn bench_create_frame(b: &mut Bencher) {
        b.iter(|| {
            let f: Frame3<IntT, FloatT, BoolT> = frame![
                Column::new((0..SIZE).map(Some)),
                Column::new((0..SIZE).map(|v| v as f64).map(Some)),
                Column::new((0..SIZE).map(|v| v % 2 == 0).map(Some)),
            ];
            f
        })
    }


    #[bench]
    fn bench_index_column_sequence(b: &mut Bencher) {
        let mut c = Column::new((0..SIZE).map(Some));
        b.iter(|| {
            c.build_index();
            c.drop_index();
        })
    }

    #[bench]
    fn bench_index_column_random_with_10_dupes(b: &mut Bencher) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut c = Column::new((0..SIZE).map(|_| rng.gen_range(0, SIZE / 10)).map(Some));
        b.iter(|| {
            c.build_index();
            c.drop_index();
        })
    }

    #[bench]
    fn bench_inner_join(b: &mut Bencher) {
        let f1 = Frame::with((0..SIZE).map(Some));
        f1.get(IntT).build_index();
        let f2 = f1.clone();
        b.iter(|| f1.clone().inner_join(&f2, IntT, IntT))
    }
}
