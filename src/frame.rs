use std::marker::PhantomData;

// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

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
    type Remainder;
    fn extract(self) -> (Vec<Target::Output>, Self::Remainder);
    fn fetch(&self) -> &[Target::Output];
}

impl<Head: Token, Tail> Fetcher<Head, Here> for HCons<Head, Tail> {
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
    let (v, h) = h.extract::<IntT, _>();
    assert_eq!(v, &[10]);
    let (v, h) = h.pop();
    assert_eq!(v, &[String::from("Hello")]);
    let (v, HNil) = h.extract::<FloatT, _>();
    assert_eq!(v, &[1.23]);
}

#[test]
fn test_concat() {
    use self::tokens::*;
    let h1: Frame2<IntT, FloatT> = HNil.addcol(vec![10i64]).addcol(vec![1.23f64]);
    let h2: Frame1<StringT> = HNil.addcol(vec![String::from("Hello")]);
    let _h3: Frame3<IntT, FloatT, StringT> = h1.concat(h2);
}
