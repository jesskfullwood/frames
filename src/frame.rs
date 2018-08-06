use std::marker::PhantomData;

use Result;

// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

pub struct Frame<H: HList> {
    hlist: H,
    pub len: usize,
}

impl Frame<HNil> {
    fn new() -> Self {
        Frame {
            hlist: HNil,
            len: 0,
        }
    }
}

impl<H: HList> Frame<H> {
    pub fn addcol<T: Token>(self, h: Vec<T::Output>) -> Result<Frame<HCons<T, H>>> {
        if !H::IS_ROOT && h.len() != self.len {
            bail!("Mismatched length")
        } else {
            Ok(Frame {
                len: h.len(),
                hlist: self.hlist.addcol(h),
            })
        }
    }

    #[inline(always)]
    pub fn fetch<T: Token, Index>(&self) -> &[T::Output]
    where
        H: Fetcher<T, Index>,
    {
        Fetcher::fetch(&self.hlist)
    }

    #[inline(always)]
    pub fn extract<T: Token, Index>(
        self,
    ) -> (Vec<T::Output>, Frame<<H as Fetcher<T, Index>>::Remainder>)
    where
        H: Fetcher<T, Index>,
    {
        let (v, hlist) = Fetcher::extract(self.hlist);
        (
            v,
            Frame {
                hlist,
                len: self.len,
            },
        )
    }

    pub fn concat<J: Concat<Self>>(self, other: J) -> J::Combined {
        other.concat_front(self)
    }

    pub fn num_cols(&self) -> usize {
        H::SIZE
    }
}

impl<Head: Token, Tail: HList> Frame<HCons<Head, Tail>> {
    #[inline(always)]
    pub fn pop(self) -> (Vec<Head::Output>, Frame<Tail>) {
        let tail = Frame {
            hlist: self.hlist.tail,
            len: self.len,
        };
        (self.hlist.head, tail)
    }
}

pub trait HList: Sized {
    const SIZE: usize;
    const IS_ROOT: bool;

    #[inline]
    fn size(&self) -> usize {
        Self::SIZE
    }

    fn addcol<T: Token>(self, head: Vec<T::Output>) -> HCons<T, Self> {
        HCons {
            head: head,
            tail: self,
        }
    }
}

pub trait Token {
    type Output;
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct HNil;

impl HList for HNil {
    const SIZE: usize = 0;
    const IS_ROOT: bool = true;
}

pub struct HCons<H: Token, T> {
    pub head: Vec<H::Output>,
    pub tail: T,
}

impl<H: Token, T: HList> HList for HCons<H, T> {
    const SIZE: usize = 1 + <T as HList>::SIZE;
    const IS_ROOT: bool = false;
}

impl<Head: Token, Tail> HCons<Head, Tail> {
    #[inline(always)]
    pub fn extract<T: Token, Index>(
        self,
    ) -> (Vec<T::Output>, <Self as Fetcher<T, Index>>::Remainder)
    where
        Self: Fetcher<T, Index>,
    {
        Fetcher::extract(self)
    }

    #[inline(always)]
    pub fn fetch<T: Token, Index>(&self) -> &[T::Output]
    where
        Self: Fetcher<T, Index>,
    {
        Fetcher::fetch(self)
    }

    #[inline(always)]
    pub fn pop(self) -> (Vec<Head::Output>, Tail) {
        (self.head, self.tail)
    }

    fn concat<J: Concat<Self>>(self, other: J) -> J::Combined {
        other.concat_front(self)
    }
}

pub trait Concat<J> {
    type Combined;
    fn concat_front(self, other: J) -> Self::Combined;
}

impl<J> Concat<J> for HNil {
    type Combined = J;
    fn concat_front(self, other: J) -> Self::Combined {
        other
    }
}

impl<Head, Tail, J> Concat<J> for HCons<Head, Tail>
where
    Head: Token,
    Tail: Concat<J>,
{
    type Combined = HCons<Head, <Tail as Concat<J>>::Combined>;
    fn concat_front(self, other: J) -> Self::Combined {
        HCons {
            head: self.head,
            tail: self.tail.concat_front(other),
        }
    }
}

pub trait Fetcher<Target: Token, Index> {
    type Remainder: HList;
    fn extract(self) -> (Vec<Target::Output>, Self::Remainder);
    fn fetch(&self) -> &[Target::Output];
}

impl<Head: Token, Tail: HList> Fetcher<Head, Here> for HCons<Head, Tail> {
    type Remainder = Tail;

    fn extract(self) -> (Vec<Head::Output>, Self::Remainder) {
        (self.head, self.tail)
    }
    fn fetch(&self) -> &[Head::Output] {
        &self.head
    }
}

impl<Head, Tail, FromTail, TailIndex> Fetcher<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: Token,
    FromTail: Token,
    Tail: Fetcher<FromTail, TailIndex>,
{
    type Remainder = HCons<Head, <Tail as Fetcher<FromTail, TailIndex>>::Remainder>;

    fn extract(self) -> (Vec<FromTail::Output>, Self::Remainder) {
        let (target, tail_remainder): (
            Vec<FromTail::Output>,
            <Tail as Fetcher<FromTail, TailIndex>>::Remainder,
        ) = <Tail as Fetcher<FromTail, TailIndex>>::extract(self.tail);
        (
            target,
            HCons {
                head: self.head,
                tail: tail_remainder,
            },
        )
    }
    fn fetch(&self) -> &[FromTail::Output] {
        <Tail as Fetcher<FromTail, TailIndex>>::fetch(&self.tail)
    }
}

pub struct Here {
    _priv: (),
}

pub struct There<T> {
    _marker: PhantomData<T>,
}

macro_rules! coldef {
    ($name:ident, $typ:ty) => {
        struct $name;
        impl Token for $name {
            type Output = $typ;
        }
    };
}

type Frame0 = Frame<HNil>;
type Frame1<T1> = HCons<T1, HNil>;
type Frame2<T1, T2> = HCons<T2, Frame1<T1>>;
type Frame3<T1, T2, T3> = HCons<T3, Frame2<T1, T2>>;

#[cfg(test)]
mod tests {

    use super::*;

    coldef!(IntT, i64);
    coldef!(StringT, String);
    coldef!(FloatT, f64);

    type Data3 = Frame3<IntT, FloatT, StringT>;

    #[test]
    fn test_basic_frame() -> Result<()> {
        let f: Frame<Data3> = Frame::new()
            .addcol(vec![10i64])?
            .addcol(vec![1.23f64])?
            .addcol(vec![String::from("Hello")])?;
        assert_eq!(f.num_cols(), 3);
        assert_eq!(f.len, 1);
        {
            let f = f.fetch::<FloatT, _>();
            assert_eq!(f, &[1.23]);
        }
        let (v, f) = f.extract::<IntT, _>();
        assert_eq!(v, &[10]);
        let (v, f) = f.pop();
        assert_eq!(v, &[String::from("Hello")]);
        let (v, _): (_, Frame0) = f.extract::<FloatT, _>();
        assert_eq!(v, &[1.23]);
        Ok(())
    }

    #[test]
    fn test_double_insert() {
        type First = There<Here>;
        let h: Frame2<IntT, IntT> = HNil.addcol(vec![10]).addcol(vec![1]);
        let (v, _) = h.extract::<IntT, First>();
        assert_eq!(v, &[10]);
    }

    #[test]
    fn test_add_col() {
        let h: Frame3<IntT, FloatT, StringT> = HNil
            .addcol(vec![10i64])
            .addcol(vec![1.23f64])
            .addcol(vec![String::from("Hello")]);
        coldef!(Added, i64);
        let h = h.addcol::<Added>(vec![123]);
        let v = h.fetch::<Added, _>();
        assert_eq!(v, &[123]);
    }

    #[test]
    fn test_concat() {
        let h1: Frame2<IntT, FloatT> = HNil.addcol(vec![10i64]).addcol(vec![1.23f64]);
        let h2: Frame1<StringT> = HNil.addcol(vec![String::from("Hello")]);
        let _h3: Frame3<IntT, FloatT, StringT> = h1.concat(h2);
    }
}
