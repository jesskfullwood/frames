pub use frame_alias::*;
use std::hash::Hash;
use std::marker::PhantomData;

use {Collection, Mask, Result};

// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

#[derive(Debug, Clone, PartialEq)]
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

    pub fn num_cols(&self) -> usize {
        H::SIZE
    }

    #[inline]
    pub fn get<Col, Index>(&self) -> &Collection<Col::Output>
    where
        Col: ColId,
        H: Selector<Col, Index>,
    {
        Selector::get(&self.hlist)
    }

    // TODO: alternative would be to explicitly pass the token
    pub fn addcol<T, I>(self, coll: I) -> Result<Frame<HCons<T, H>>>
    where
        T: ColId,
        I: Into<Collection<T::Output>>,
    {
        let coll = coll.into();
        if !H::IS_ROOT && coll.len() != self.len {
            bail!("Mismatched lengths ({} and {})", self.len(), coll.len())
        } else {
            Ok(Frame {
                len: coll.len(),
                hlist: self.hlist.addcol(coll),
            })
        }
    }

    #[inline(always)]
    pub fn extract<T, Index>(
        self,
    ) -> (
        Collection<T::Output>,
        Frame<<H as Extractor<T, Index>>::Remainder>,
    )
    where
        T: ColId,
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
}

impl<H> Frame<H>
where
    H: HList + HListExt,
{
    pub fn inner_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: Frame<Oth>,
    ) -> Frame<<<Oth as Extractor<RCol, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList + Selector<RCol, RIx> + Extractor<RCol, RIx> + Concat<H> + HListExt,
        <Oth as Extractor<RCol, RIx>>::Remainder: Concat<H>,
        LCol: ColId,
        LCol::Output: Eq + Clone + Hash,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<LCol, LIx>,
    {
        let left = self.get::<LCol, _>();
        let right = other.get::<RCol, _>();
        let (leftixs, rightixs) = left.inner_join_locs(right);
        let leftframe = self.copy_locs(&leftixs);
        let rightframe = other.copy_locs(&rightixs);
        let (_, rightframe) = rightframe.extract::<RCol, _>();
        leftframe.concat(rightframe).unwrap()
    }

    fn copy_locs(&self, locs: &[usize]) -> Frame<H> {
        Frame {
            hlist: self.hlist.copy_locs(locs),
            len: locs.len(),
        }
    }

    pub fn apply_mask(&self, mask: &Mask) -> Result<Self> {
        if mask.len() != self.len() {
            bail!("Mismatched lengths ({} and {})", self.len(), mask.len())
        }
        Ok(Frame {
            hlist: self.hlist.apply_mask(mask),
            len: mask.true_count,
        })
    }

    pub fn groupby<Col, Index>(self) -> GroupBy<H, HCons<Col, HNil>>
    where
        Col: ColId,
        Col::Output: Eq + Clone + Hash,
        H: Selector<Col, Index>,
    {
        let (grouping_index, grouped_col) = {
            // lifetimes workaround
            let grouping_col = self.get::<Col, _>();
            let mut index = grouping_col.index_values();
            index.sort_unstable();
            let groupedcol = grouping_col.copy_first_locs(&index);
            (index, groupedcol)
        };
        let grouped_frame = Frame::new().addcol(grouped_col).unwrap();
        GroupBy {
            frame: self,
            grouping_index,
            grouped_frame,
        }
    }
}

impl<Head: ColId, Tail: HList> Frame<HCons<Head, Tail>> {
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

    fn addcol<T: ColId>(self, head: impl Into<Collection<T::Output>>) -> HCons<T, Self> {
        HCons {
            head: head.into(),
            tail: self,
        }
    }
}

impl HListExt for HNil {
    fn copy_locs(&self, _: &[usize]) -> Self {
        HNil
    }

    fn apply_mask(&self, _mask: &Mask) -> Self {
        HNil
    }
}

pub trait ColId {
    type Output;
}

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct HNil;

impl HList for HNil {
    const SIZE: usize = 0;
    const IS_ROOT: bool = true;
}

#[derive(PartialEq, Debug, Clone)]
pub struct HCons<H: ColId, T> {
    pub head: Collection<H::Output>,
    pub tail: T,
}

impl<Head, Tail> HList for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList,
{
    const SIZE: usize = 1 + <Tail as HList>::SIZE;
    const IS_ROOT: bool = false;
}

pub trait HListExt {
    fn copy_locs(&self, locs: &[usize]) -> Self;
    fn apply_mask(&self, mask: &Mask) -> Self;
}

impl<Head, Tail> HListExt for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: Clone,
    Tail: HListExt,
{
    fn copy_locs(&self, locs: &[usize]) -> Self {
        HCons {
            head: self.head.copy_locs(locs),
            tail: self.tail.copy_locs(locs),
        }
    }

    fn apply_mask(&self, mask: &Mask) -> Self {
        HCons {
            head: self.head.apply_mask(mask),
            tail: self.tail.apply_mask(mask),
        }
    }
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
    Head: ColId,
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

pub trait Selector<S: ColId, I> {
    fn get(&self) -> &Collection<S::Output>;
}

impl<T: ColId, Tail> Selector<T, Here> for HCons<T, Tail> {
    fn get(&self) -> &Collection<T::Output> {
        &self.head
    }
}

impl<Head, Tail, FromTail, TailIndex> Selector<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
    Tail: Selector<FromTail, TailIndex>,
{
    fn get(&self) -> &Collection<FromTail::Output> {
        self.tail.get()
    }
}

pub trait Extractor<Target: ColId, Index> {
    type Remainder: HList;
    fn extract(self) -> (Collection<Target::Output>, Self::Remainder);
}

impl<Head: ColId, Tail: HList> Extractor<Head, Here> for HCons<Head, Tail> {
    type Remainder = Tail;

    fn extract(self) -> (Collection<Head::Output>, Self::Remainder) {
        (self.head, self.tail)
    }
}

impl<Head, Tail, FromTail, TailIndex> Extractor<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
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

pub struct GroupBy<H: HList, G: HList> {
    frame: Frame<H>,
    grouping_index: Vec<Vec<usize>>,
    grouped_frame: Frame<G>,
}

impl<H, G> GroupBy<H, G>
where
    H: HList,
    G: HList,
{
    pub fn acc<Col, NewCol, Index, AccFn>(self, func: AccFn) -> GroupBy<H, HCons<NewCol, G>>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<Col, Index>,
        AccFn: Fn(&[&Col::Output]) -> NewCol::Output,
    {
        let res: Vec<NewCol::Output> = {
            let grouped_col = self.frame.get::<Col, _>();
            let col_data = grouped_col.data();
            self.grouping_index
                .iter()
                .map(|grp| {
                    // TODO could this be done with an iterator instead of allocating a vec?
                    let to_acc: Vec<_> = grp.iter().map(|&ix| &col_data[ix]).collect();
                    func(&to_acc)
                }).collect()
        };
        let grouped_frame = self.grouped_frame.addcol(res).unwrap();
        GroupBy {
            frame: self.frame,
            grouping_index: self.grouping_index,
            grouped_frame,
        }
    }

    pub fn done(self) -> Frame<G> {
        self.grouped_frame
    }
}

pub struct Here {
    _priv: (),
}

pub struct There<T> {
    _marker: PhantomData<T>,
}

#[macro_export]
macro_rules! define_col {
    ($name:ident, $typ:ty) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        struct $name;
        impl ColId for $name {
            type Output = $typ;
        }
    };
}

#[cfg(test)]
mod tests {

    use super::*;

    define_col!(IntT, i64);
    define_col!(IntT2, i64);
    define_col!(StringT, String);
    define_col!(FloatT, f64);
    define_col!(BoolT, bool);

    type Data3 = Frame3<IntT, FloatT, StringT>;

    fn quickframe() -> Data3 {
        Frame::new()
            .addcol(vec![1, 2, 3, 4])
            .unwrap()
            .addcol(vec![5., 4., 3., 2.])
            .unwrap()
            .addcol(
                "this is the words"
                    .split(' ')
                    .map(String::from)
                    .collect::<Vec<_>>(),
            ).unwrap()
    }

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
        define_col!(Added, i64);
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
            let _f3: Frame0 = f1.concat(f2)?;
        }

        {
            let f1: Frame2<IntT, FloatT> =
                Frame::new().addcol(vec![10i64])?.addcol(vec![1.23f64])?;
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

    #[test]
    fn test_inner_join() -> Result<()> {
        let f1 = quickframe();
        let f2: Frame2<IntT2, BoolT> = Frame::new()
            .addcol(vec![3, 2, 4, 2, 1])?
            .addcol(vec![true, false, true, true, false])?;
        let f3: Frame4<IntT, FloatT, StringT, BoolT> = f1.inner_join::<IntT, IntT2, _, _, _>(f2);
        assert_eq!(f3.get::<IntT, _>(), &[1, 2, 2, 3, 4]);
        // TODO: Note sure this ordering can be relied upon
        assert_eq!(f3.get::<BoolT, _>(), &[false, false, true, true, true]);
        Ok(())
    }

    #[test]
    fn test_mask() -> Result<()> {
        let f = quickframe();
        // TODO document - keep if true or discard-if-true? At moment it's keep-if-true
        let mask = f.get::<IntT, _>().mask(|&v| v > 2);
        let f2 = f.apply_mask(&mask)?;
        assert_eq!(f2.get::<IntT, _>(), &[3, 4]);
        // Fails with incorrect len
        let mask2 = f2.get::<IntT, _>().mask(|&v| v > 2);
        assert!(f.apply_mask(&mask2).is_err());
        Ok(())
    }

    #[test]
    fn test_groupby() -> Result<()> {
        // TODO special method for first column, or some kind of convenience builder
        let f: Frame2<IntT, FloatT> = Frame::new()
            .addcol(vec![1, 3, 2, 3, 4, 2])?
            .addcol(vec![1., 2., 1., 1., 1., 1.])?;
        define_col!(FloatSum, f64);
        // TODO can we rewrite to get rid of the dangling type parameters?
        let g = f
            .groupby::<IntT, _>()
            .acc::<FloatT, FloatSum, _, _>(|slice| slice.iter().map(|v| *v).sum())
            .done();
        assert_eq!(g.get::<IntT, _>(), &[1, 3, 2, 4]);
        assert_eq!(g.get::<FloatSum, _>(), &[1., 3., 2., 1.]);
        Ok(())
    }
}
