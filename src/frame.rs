use std::marker::PhantomData;

use {Collection, Result};

// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

pub struct Frame<H: HList> {
    hlist: H,
    len: usize,
}

impl Frame<HNil> {
    pub fn new() -> Self {
        Frame {
            hlist: HNil,
            len: 0,
        }
    }
}

impl<H: HList> Frame<H> {
    fn len(&self) -> usize {
        self.len
    }

    // TODO: alternative would be to explicitly pass the token
    pub fn addcol<T, I>(self, coll: I) -> Result<Frame<HCons<T, H>>>
    where
        T: Token,
        I: Into<Collection<T::Output>>,
    {
        let coll = coll.into();
        if !H::IS_ROOT && coll.len() != self.len {
            bail!("Mismatched length")
        } else {
            Ok(Frame {
                len: coll.len(),
                hlist: self.hlist.addcol(coll),
            })
        }
    }

    #[inline(always)]
    pub fn get<T, Index>(&self) -> &Collection<T::Output>
    where
        T: Token,
        H: Selector<T, Index>,
    {
        Selector::get(&self.hlist)
    }

    #[inline(always)]
    pub fn extract<T, Index>(
        self,
    ) -> (
        Collection<T::Output>,
        Frame<<H as Extractor<T, Index>>::Remainder>,
    )
    where
        T: Token,
        H: Extractor<T, Index>,
    {
        let (v, hlist) = Extractor::extract(self.hlist);
        (
            v,
            Frame {
                hlist,
                len: self.len,
            },
        )
    }

    pub fn concat<C: HList + Concat<H>>(self, other: Frame<C>) -> Result<Frame<C::Combined>> {
        other.concat_front(self)
    }

    fn concat_front<C: HList>(self, other: Frame<C>) -> Result<Frame<H::Combined>>
    where
        H: Concat<C>,
    {
        let len = if H::IS_ROOT {
            other.len()
        } else if C::IS_ROOT {
            self.len()
        } else if self.len() != other.len() {
            bail!("Mismatched lengths ({} and {})", other.len(), self.len())
        } else {
            self.len()
        };
        Ok(Frame {
            len,
            hlist: self.hlist.concat_front(other.hlist),
        })
    }

    pub fn num_cols(&self) -> usize {
        H::SIZE
    }
}

impl<Head: Token, Tail: HList> Frame<HCons<Head, Tail>> {
    #[inline(always)]
    pub fn pop(self) -> (Collection<Head::Output>, Frame<Tail>) {
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

    fn addcol<T: Token>(self, head: impl Into<Collection<T::Output>>) -> HCons<T, Self> {
        HCons {
            head: head.into(),
            tail: self,
        }
    }
}

pub trait Token {
    type Output;
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct HNil;

impl HList for HNil {
    const SIZE: usize = 0;
    const IS_ROOT: bool = true;
}

#[derive(PartialEq, Debug, Clone)]
pub struct HCons<H: Token, T> {
    pub head: Collection<H::Output>,
    pub tail: T,
}

impl<H: Token, T: HList> HList for HCons<H, T> {
    const SIZE: usize = 1 + <T as HList>::SIZE;
    const IS_ROOT: bool = false;
}

pub trait Concat<C> {
    type Combined: HList;
    fn concat_front(self, other: C) -> Self::Combined;
}

impl<C: HList> Concat<C> for HNil {
    type Combined = C;
    fn concat_front(self, other: C) -> Self::Combined {
        other
    }
}

impl<Head, Tail, C> Concat<C> for HCons<Head, Tail>
where
    Head: Token,
    Tail: Concat<C>,
{
    type Combined = HCons<Head, <Tail as Concat<C>>::Combined>;
    fn concat_front(self, other: C) -> Self::Combined {
        HCons {
            head: self.head,
            tail: self.tail.concat_front(other),
        }
    }
}

pub trait Selector<S: Token, I> {
    fn get(&self) -> &Collection<S::Output>;
}

impl<T: Token, Tail> Selector<T, Here> for HCons<T, Tail> {
    fn get(&self) -> &Collection<T::Output> {
        &self.head
    }
}

impl<Head, Tail, FromTail, TailIndex> Selector<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: Token,
    FromTail: Token,
    Tail: Selector<FromTail, TailIndex>,
{
    fn get(&self) -> &Collection<FromTail::Output> {
        self.tail.get()
    }
}

pub trait Extractor<Target: Token, Index> {
    type Remainder: HList;
    fn extract(self) -> (Collection<Target::Output>, Self::Remainder);
}

impl<Head: Token, Tail: HList> Extractor<Head, Here> for HCons<Head, Tail> {
    type Remainder = Tail;

    fn extract(self) -> (Collection<Head::Output>, Self::Remainder) {
        (self.head, self.tail)
    }
}

impl<Head, Tail, FromTail, TailIndex> Extractor<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: Token,
    FromTail: Token,
    Tail: Extractor<FromTail, TailIndex>,
{
    type Remainder = HCons<Head, <Tail as Extractor<FromTail, TailIndex>>::Remainder>;

    fn extract(self) -> (Collection<FromTail::Output>, Self::Remainder) {
        let (target, tail_remainder): (
            Collection<FromTail::Output>,
            <Tail as Extractor<FromTail, TailIndex>>::Remainder,
        ) = <Tail as Extractor<FromTail, TailIndex>>::extract(self.tail);
        (
            target,
            HCons {
                head: self.head,
                tail: tail_remainder,
            },
        )
    }
}

pub struct Here {
    _priv: (),
}

pub struct There<T> {
    _marker: PhantomData<T>,
}

#[macro_export]
macro_rules! coldef {
    ($name:ident, $typ:ty) => {
        struct $name;
        impl Token for $name {
            type Output = $typ;
        }
    };
}

type List1<T1> = HCons<T1, HNil>;
type List2<T1, T2> = HCons<T2, List1<T1>>;
type List3<T1, T2, T3> = HCons<T3, List2<T1, T2>>;
pub type Frame0 = Frame<HNil>;
pub type Frame1<T1> = Frame<List1<T1>>;
pub type Frame2<T1, T2> = Frame<List2<T1, T2>>;
pub type Frame3<T1, T2, T3> = Frame<List3<T1, T2, T3>>;

#[cfg(test)]
mod tests {

    use super::*;

    coldef!(IntT, i64);
    coldef!(StringT, String);
    coldef!(FloatT, f64);

    type Data3 = Frame3<IntT, FloatT, StringT>;

    #[test]
    fn test_basic_frame() -> Result<()> {
        let f: Data3 = Frame::new()
            .addcol(vec![10i64])?
            .addcol(vec![1.23f64])?
            .addcol(vec![String::from("Hello")])?;
        assert_eq!(f.num_cols(), 3);
        assert_eq!(f.len, 1);
        {
            let f = f.get::<FloatT, _>();
            assert_eq!(f, &[1.23f64])
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
    fn test_double_insert() -> Result<()> {
        type First = There<Here>;
        let f: Frame2<IntT, IntT> = Frame::new().addcol(vec![10])?.addcol(vec![1])?;
        let (v, _) = f.extract::<IntT, First>();
        assert_eq!(v, &[10]);
        Ok(())
    }

    #[test]
    fn test_add_coldef() -> Result<()> {
        let f: Data3 = Frame::new()
            .addcol(vec![10i64])?
            .addcol(vec![1.23f64])?
            .addcol(vec![String::from("Hello")])?;
        coldef!(Added, i64);
        let f = f.addcol::<Added, _>(vec![123])?;
        let v = f.get::<Added, _>();
        assert_eq!(v, &[123]);
        Ok(())
    }

    #[test]
    fn test_concat() -> Result<()> {
        {
            let f1 = Frame::new();
            let f2 = Frame::new();
            let f3: Frame0 = f1.concat(f2)?;
        }

        {
            let f1: Frame2<IntT, FloatT> = Frame::new().addcol(vec![10i64])?.addcol(vec![1.23f64])?;
            let f2: Frame1<StringT> = Frame::new().addcol(vec![String::from("Hello")])?;
            let _f3: Frame3<IntT, FloatT, StringT> = f1.concat(f2)?;
        }

        {
            let f1 = Frame::new();
            let f2 = Frame::new().addcol::<IntT, _>(vec![1, 2, 3])?;
            let f3 = Frame::new().addcol::<FloatT, _>(vec![1., 2.])?;
            let f4: Frame1<_> = f1.concat(f2)?;
            assert!(f4.concat(f3).is_err()) // mismatched types
        }

        Ok(())
    }
}
