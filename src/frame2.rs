// pub use frunk::hlist::{HList, HNil, HCons, Plucker, Selector};

use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
struct Frame<H>
where
    H: HList,
{
    pub(crate) inner: H,
    pub(crate) len: usize,
}

impl<H> Frame<H>
where
    H: HList + Freezable,
    H::Frozen: HList,
{
    fn freeze(self) -> Frame<H::Frozen> {
        Frame {
            inner: self.inner.freeze(),
            len: self.len,
        }
    }
}

trait HList {}

struct Nil;
impl HList for Nil {}
impl Freezable for Nil {
    type Frozen = Self;
    fn freeze(self) -> Self::Frozen {
        self
    }
}

struct Cons<Head: Column, Tail: HList> {
    head: Head,
    tail: Tail,
}

impl<Head: Column, Tail: HList> HList for Cons<Head, Tail> {}

trait Column {
    type T;
    fn iter_opt<'a>(&'a self) -> Box<Iterator<Item = Option<&'a Self::T>> + 'a>;
}

impl<T> Column for Vec<T> {
    type T = T;
    fn iter_opt<'a>(&'a self) -> Box<Iterator<Item = Option<&'a Self::T>> + 'a> {
        Box::new(self.iter().map(Some))
    }
}

impl<T> Column for Arc<Vec<T>> {
    type T = T;
    fn iter_opt<'a>(&'a self) -> Box<Iterator<Item = Option<&'a Self::T>> + 'a> {
        Box::new(self.iter().map(Some))
    }
}

trait Freezable {
    type Frozen;
    fn freeze(self) -> Self::Frozen;
}

impl<T> Freezable for Vec<T> {
    type Frozen = Arc<Vec<T>>;
    fn freeze(self) -> Self::Frozen {
        Arc::new(self)
    }
}

impl<T> Freezable for Arc<Vec<T>> {
    type Frozen = Self;
    fn freeze(self) -> Self::Frozen {
        self
    }
}

impl<Head, Tail> Freezable for Cons<Head, Tail>
where
    Head: Column + Freezable,
    Head::Frozen: Column,
    Tail: Freezable + HList,
    Tail::Frozen: HList,
{
    type Frozen = Cons<Head::Frozen, Tail::Frozen>;
    fn freeze(self) -> Self::Frozen {
        unimplemented!()
    }
}

fn quickframe() -> Frame<Cons<Vec<u32>, Cons<Arc<Vec<u32>>, Nil>>> {
    let frame = Frame {
        inner: Cons {
            head: vec![1, 2, 3],
            tail: Cons {
                head: Arc::new(vec![4, 5, 6]),
                tail: Nil,
            },
        },
        len: 3,
    };
    frame
}

fn nothing(_: u32) {}

fn test() {
    let f = quickframe();
    let frozen = f.freeze();
    nothing(frozen);
}
