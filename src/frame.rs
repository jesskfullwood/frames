use std::marker::PhantomData;

// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details.

pub trait HList: Sized {
    const LEN: usize;

    #[inline]
    fn len(&self) -> usize {
        Self::LEN
    }

    fn addcol<H: Token>(self, h: Vec<H::Output>) -> HCons<H, Self> {
        HCons {
            head: h,
            tail: self,
        }
    }
}

pub trait Token {
    type Output;
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HNil;

impl HList for HNil {
    const LEN: usize = 0;
}

pub struct HCons<H: Token, T> {
    pub head: Vec<H::Output>,
    pub tail: T,
}

impl<H: Token, T: HList> HList for HCons<H, T> {
    const LEN: usize = 1 + <T as HList>::LEN;
}

impl<Head: Token, Tail> HCons<Head, Tail> {
    #[inline(always)]
    pub fn remove<T: Token, Index>(self) -> (Vec<T::Output>, <Self as Fetcher<T, Index>>::Remainder)
    where
        Self: Fetcher<T, Index>,
    {
        Fetcher::remove(self)
    }
    #[inline(always)]
    pub fn fetch<T: Token, Index>(&self) -> &[T::Output]
    where
        Self: Fetcher<T, Index>,
    {
        Fetcher::fetch(self)
    }
}

pub trait Fetcher<Target: Token, Index> {
    type Remainder;
    fn remove(self) -> (Vec<Target::Output>, Self::Remainder);
    fn fetch(&self) -> &[Target::Output];
}

impl<Head: Token, Tail> Fetcher<Head, Here> for HCons<Head, Tail> {
    type Remainder = Tail;

    fn remove(self) -> (Vec<Head::Output>, Self::Remainder) {
        (self.head, self.tail)
    }
    fn fetch(&self) -> &[Head::Output] {
        &self.head
    }
}

impl<Head: Token, Tail, FromTail: Token, TailIndex> Fetcher<FromTail, There<TailIndex>>
    for HCons<Head, Tail>
where
    Tail: Fetcher<FromTail, TailIndex>,
{
    type Remainder = HCons<Head, <Tail as Fetcher<FromTail, TailIndex>>::Remainder>;

    fn remove(self) -> (Vec<FromTail::Output>, Self::Remainder) {
        let (target, tail_remainder): (
            Vec<FromTail::Output>,
            <Tail as Fetcher<FromTail, TailIndex>>::Remainder,
        ) = <Tail as Fetcher<FromTail, TailIndex>>::remove(self.tail);
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

mod tokens {
    use super::Token;

    pub struct IntT;
    pub struct StringT;
    pub struct FloatT;

    impl Token for FloatT {
        type Output = f64;
    }

    impl Token for IntT {
        type Output = i64;
    }

    impl Token for StringT {
        type Output = ::std::string::String;
    }
}

type Frame1<T1> = HCons<T1, HNil>;
type Frame2<T1, T2> = HCons<T2, Frame1<T1>>;
type Frame3<T1, T2, T3> = HCons<T3, Frame2<T1, T2>>;

#[test]
fn test_basic_frame() {
    use self::tokens::*;
    let h: Frame3<IntT, FloatT, StringT> = HNil.addcol(vec![10i64])
        .addcol(vec![1.23f64])
        .addcol(vec![String::from("Hello")]);
    assert_eq!(h.len(), 3);
    {
        let f = h.fetch::<FloatT, _>();
        assert_eq!(f, &[1.23]);
    }
    let (v, h) = h.remove::<IntT, _>();
    assert_eq!(v, &[10]);
    let (v, h) = h.remove::<StringT, _>();
    assert_eq!(v, &[String::from("Hello")]);
    let (v, HNil) = h.remove::<FloatT, _>();
    assert_eq!(v, &[1.23]);
}

#[test]
fn test_join() {
    // let h: Frame3<IntT, FloatT, StringT> = HNil.addcol(10i64)
    //     .addcol(1.23f64)
    //     .addcol(String::from("Hello"));

}
