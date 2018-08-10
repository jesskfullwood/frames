pub use frame_alias::*;
use id;
use std::hash::Hash;
use std::marker::PhantomData;

use {Collection, Mask, Result};
// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

// TODO tag everything with #[must_use]

pub trait ColId {
    const NAME: &'static str;
    type Output;
}

// ### Frame ###

#[derive(Debug, Clone, PartialEq)]
pub struct Frame<H: HList> {
    pub(crate) hlist: H,
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

impl Default for Frame<HNil> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: HList> Frame<H> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_cols(&self) -> usize {
        H::SIZE
    }

    pub fn names(&self) -> Vec<&'static str> {
        unimplemented!();
    }

    #[inline]
    pub fn get<Col, Index>(&self) -> &Collection<Col::Output>
    where
        Col: ColId,
        H: Selector<Col, Index>,
    {
        Selector::get(&self.hlist)
    }

    // TODO: alternative would be to explicitly pass the Col token
    // TODO: what would be nicer is a `setcol` func which either adds or modifies
    pub fn addcol<Col, Data>(self, coll: Data) -> Result<Frame<HCons<Col, H>>>
    where
        Col: ColId,
        Data: Into<Collection<Col::Output>>,
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

    pub fn replace<Col: ColId, Index>(&mut self, newcol: Collection<Col::Output>)
    where
        H: Replacer<Col, Index>,
    {
        self.hlist.replace(newcol)
    }

    pub fn map_replace<Col, NewCol, Index, F>(
        self,
        func: F,
    ) -> Frame<<H as Mapper<Col, NewCol, Index>>::Mapped>
    where
        Col: ColId,
        NewCol: ColId,
        H: Mapper<Col, NewCol, Index>,
        F: Fn(Option<&Col::Output>) -> Option<NewCol::Output>,
    {
        Frame {
            hlist: self.hlist.map_replace(func),
            len: self.len,
        }
    }

    pub fn map_replace_notnull<Col, NewCol, Index, F>(
        self,
        func: F,
    ) -> Frame<<H as Mapper<Col, NewCol, Index>>::Mapped>
    where
        Col: ColId,
        NewCol: ColId,
        H: Mapper<Col, NewCol, Index>,
        F: Fn(&Col::Output) -> NewCol::Output,
    {
        Frame {
            hlist: self.hlist.map_replace_notnull(func),
            len: self.len,
        }
    }

    #[inline(always)]
    pub fn extract<Col, Index>(
        self,
    ) -> (
        Collection<Col::Output>,
        Frame<<H as Extractor<Col, Index>>::Remainder>,
    )
    where
        Col: ColId,
        H: Extractor<Col, Index>,
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
    H: HListExt,
{
    pub fn inner_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
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

    pub fn left_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
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
        let (leftixs, rightixs) = left.left_join_locs(right);
        let leftframe = self.copy_locs(&leftixs);
        let rightframe = other.copy_locs_opt(&rightixs);
        let (_, rightframe) = rightframe.extract::<RCol, _>();
        leftframe.concat(rightframe).unwrap()
    }

    pub fn outer_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
    ) -> Frame<<<Oth as Extractor<RCol, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList + Selector<RCol, RIx> + Extractor<RCol, RIx> + Concat<H> + HListExt,
        <Oth as Extractor<RCol, RIx>>::Remainder: Concat<H>,
        LCol: ColId,
        LCol::Output: Eq + Clone + Hash,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<LCol, LIx> + Replacer<LCol, LIx>,
    {
        let left = self.get::<LCol, _>();
        let right = other.get::<RCol, _>();
        let (leftixs, rightixs) = left.outer_join_locs(right);
        let mut leftframe = self.copy_locs_opt(&leftixs);
        let rightframe = other.copy_locs_opt(&rightixs);
        let (rjoined, rightframe) = rightframe.extract::<RCol, _>();
        let joined = {
            let ljoined = leftframe.get::<LCol, _>();
            Collection::new_opt(
                ljoined
                    .iter()
                    .zip(rjoined.iter())
                    .map(|(left, right)| left.or(right).cloned()),
            )
        };
        leftframe.replace(joined);
        leftframe.concat(rightframe).unwrap()
    }

    fn copy_locs(&self, locs: &[usize]) -> Frame<H> {
        Frame {
            hlist: self.hlist.copy_locs(locs),
            len: locs.len(),
        }
    }

    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Frame<H> {
        Frame {
            hlist: self.hlist.copy_locs_opt(locs),
            len: locs.len(),
        }
    }

    pub fn filter<Col, Index, F>(&self, func: F) -> Self
    where
        Col: ColId,
        F: Fn(&Col::Output) -> bool,
        H: Selector<Col, Index>,
    {
        // TODO also add filter2, filter3...
        let mask = self.get::<Col, _>().mask(func);
        // We know the mask is the right length
        self.filter_mask(&mask).unwrap()
    }

    pub fn filter_mask(&self, mask: &Mask) -> Result<Self> {
        if mask.len() != self.len() {
            bail!("Mismatched lengths ({} and {})", self.len(), mask.len())
        }
        Ok(Frame {
            hlist: self.hlist.filter_mask(mask),
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

impl<'a, H> Frame<H>
where
    Self: Flatten<'a>,
    H: HList + RowTuple<'a, Tuple = <Self as Flatten<'a>>::Nested>,
{
    pub fn get_row(&'a self, index: usize) -> Option<<Self as Flatten>::Flattened> {
        if index >= self.len() {
            return None;
        }
        let nested = self.hlist.get_tuple(index);
        Some(Self::unnest(nested))
    }

    pub fn iter_rows(&'a self) -> impl Iterator<Item = <Self as Flatten>::Flattened> {
        IterRows {
            frame: &self,
            index: 0,
        }
    }
}

// ### HList struct defs ###

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct HNil;

#[derive(PartialEq, Debug, Clone)]
pub struct HCons<H: ColId, T> {
    pub head: Collection<H::Output>,
    pub tail: T,
}

pub struct Here {
    _priv: (),
}

pub struct There<T> {
    _marker: PhantomData<T>,
}

// ### HList ###

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

impl HList for HNil {
    const SIZE: usize = 0;
    const IS_ROOT: bool = true;
}

impl<Head, Tail> HList for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList,
{
    const SIZE: usize = 1 + <Tail as HList>::SIZE;
    const IS_ROOT: bool = false;
}

// ### HListExt ###

pub trait HListExt: HList {
    fn copy_locs(&self, locs: &[usize]) -> Self;
    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self;
    fn filter_mask(&self, mask: &Mask) -> Self;
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

    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self {
        HCons {
            head: self.head.copy_locs_opt(locs),
            tail: self.tail.copy_locs_opt(locs),
        }
    }

    fn filter_mask(&self, mask: &Mask) -> Self {
        HCons {
            head: self.head.filter_mask(mask),
            tail: self.tail.filter_mask(mask),
        }
    }
}

impl HListExt for HNil {
    fn copy_locs(&self, _: &[usize]) -> Self {
        HNil
    }

    fn copy_locs_opt(&self, _: &[Option<usize>]) -> Self {
        HNil
    }

    fn filter_mask(&self, _mask: &Mask) -> Self {
        HNil
    }
}

// ### Concat ###

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

// ### Selector ###

pub trait Selector<S: ColId, Index> {
    fn get(&self) -> &Collection<S::Output>;
}

impl<Col: ColId, Tail> Selector<Col, Here> for HCons<Col, Tail> {
    fn get(&self) -> &Collection<Col::Output> {
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

// ### Extractor ###

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

// ### Replacer ###

pub trait Replacer<Target: ColId, Index> {
    fn replace(&mut self, newcol: Collection<Target::Output>);
}

impl<Head, Tail> Replacer<Head, Here> for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList,
{
    fn replace(&mut self, newcol: Collection<Head::Output>) {
        self.head = newcol;
    }
}

impl<Head, Tail, FromTail, TailIndex> Replacer<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
    Tail: HList + Replacer<FromTail, TailIndex>,
{
    fn replace(&mut self, newcol: Collection<FromTail::Output>) {
        self.tail.replace(newcol)
    }
}

// ### Mapper ###

pub trait Mapper<Target: ColId, NewCol: ColId, Index> {
    type Mapped: HList;
    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&Target::Output>) -> Option<<NewCol as ColId>::Output>;

    fn map_replace_notnull<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Target::Output) -> <NewCol as ColId>::Output;
}

impl<Head, Tail, NewCol> Mapper<Head, NewCol, Here> for HCons<Head, Tail>
where
    Head: ColId,
    NewCol: ColId,
    Tail: HList,
{
    type Mapped = HCons<NewCol, Tail>;
    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&Head::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        HCons {
            head: self.head.map(func),
            tail: self.tail,
        }
    }

    fn map_replace_notnull<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Head::Output) -> <NewCol as ColId>::Output,
    {
        HCons {
            head: self.head.map_notnull(func),
            tail: self.tail,
        }
    }
}

impl<Head, Tail, NewCol, FromTail, TailIndex> Mapper<FromTail, NewCol, There<TailIndex>>
    for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
    NewCol: ColId,
    Tail: HList + Mapper<FromTail, NewCol, TailIndex>,
{
    type Mapped = HCons<Head, <Tail as Mapper<FromTail, NewCol, TailIndex>>::Mapped>;

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&FromTail::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        HCons {
            head: self.head,
            tail: self.tail.map_replace(func),
        }
    }

    fn map_replace_notnull<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&FromTail::Output) -> <NewCol as ColId>::Output,
    {
        HCons {
            head: self.head,
            tail: self.tail.map_replace_notnull(func),
        }
    }
}

// ### GroupBy ###

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
    // TODO Accumulate WITHOUT nulls
    // Also need an acc WITH nulls
    pub fn acc<Col, NewCol, Index, AccFn>(self, func: AccFn) -> GroupBy<H, HCons<NewCol, G>>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<Col, Index>,
        AccFn: Fn(&[&Col::Output]) -> NewCol::Output,
    {
        let res: Vec<NewCol::Output> = {
            let grouped_col = self.frame.get::<Col, _>();
            self.grouping_index
                .iter()
                .map(|grp| {
                    // TODO could this be done with an iterator instead of allocating a vec?
                    let to_acc: Vec<&Col::Output> = grouped_col
                        .iterate_indices(grp.iter().cloned())
                        .filter_map(id)
                        .collect();
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

// ### RowTuple ###

pub trait RowTuple<'a> {
    type Tuple;
    fn get_tuple(&'a self, index: usize) -> Self::Tuple;
}

impl<'a, Head, Tail> RowTuple<'a> for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: 'a,
    Tail: RowTuple<'a>,
{
    type Tuple = (Option<&'a Head::Output>, Tail::Tuple);
    fn get_tuple(&'a self, index: usize) -> Self::Tuple {
        (self.head.get(index).unwrap(), self.tail.get_tuple(index))
    }
}

impl<'a> RowTuple<'a> for HNil {
    type Tuple = ();
    fn get_tuple(&'a self, _index: usize) -> Self::Tuple {
        ()
    }
}

// ### Flatten ###

pub trait Flatten<'a> {
    type Nested;
    type Flattened;
    fn unnest(nested: Self::Nested) -> Self::Flattened;
}

// TODO Implement this in a macro for all (many) frames
impl<'a, T1, T2, T3> Flatten<'a> for Frame3<T1, T2, T3>
where
    T1: ColId + 'a,
    T2: ColId + 'a,
    T3: ColId + 'a,
{
    type Nested = (
        Option<&'a <T3 as ColId>::Output>,
        (
            Option<&'a <T2 as ColId>::Output>,
            (Option<&'a <T1 as ColId>::Output>, ()),
        ),
    );
    type Flattened = (
        Option<&'a <T1 as ColId>::Output>,
        Option<&'a <T2 as ColId>::Output>,
        Option<&'a <T3 as ColId>::Output>,
    );

    fn unnest(nested: Self::Nested) -> Self::Flattened {
        let (opt3, (opt2, (opt1, ()))) = nested;
        (opt1, opt2, opt3)
    }
}

struct IterRows<'a, H: HList + 'a> {
    frame: &'a Frame<H>,
    index: usize,
}

impl<'a, H> Iterator for IterRows<'a, H>
where
    H: HList + RowTuple<'a, Tuple = <Frame<H> as Flatten<'a>>::Nested>,
    Frame<H>: Flatten<'a>,
{
    type Item = <Frame<H> as Flatten<'a>>::Flattened;
    fn next(&mut self) -> Option<Self::Item> {
        match (*self.frame).get_row(self.index) {
            Some(r) => {
                self.index += 1;
                Some(r)
            }
            None => None,
        }
    }
}

// trait Applyable {
//     type Ret;
//     fn apply(&self) -> Self::Ret;
// }

// trait Applyer<F> {
//     type Ret;
//     fn apply(&self) -> Vec<Self::Ret>;
// }

// impl<Head, Tail, F> Applyer<F> for HCons<Head, Tail>
// where
//     Head: ColId,
//     Collection<Head::Output>: Applyable,
//     Tail: HList + Applyer<F>,
// {
//     type Ret = F::Ret;
//     fn apply(&self) -> Vec<F::Ret> {
//         let ret = self.tail.apply();
//         ret.push(self.head.apply());
//         ret
//     }
// }

// impl<W: WriteToBuffer> Applyer for W {
//     type Ret = (Vec<u8>, Vec<usize>);
//     fn apply(&self) -> Self::Ret {
//         self.write_to_buffer()
//     }
// }

#[macro_export]
macro_rules! define_col {
    ($tyname:ident, $typ:ty) => {
        define_col!($tyname, $typ, $tyname);
    };
    ($tyname:ident, $typ:ty, $name:ident) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $name;
        impl ColId for $name {
            const NAME: &'static str = stringify!($name);
            type Output = $typ;
        }
    };
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    define_col!(IntT, i64);
    define_col!(StringT, String);
    define_col!(FloatT, f64);
    define_col!(BoolT, bool);

    pub(crate) type Data3 = Frame3<IntT, FloatT, StringT>;

    pub(crate) fn quickframe() -> Data3 {
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

}

#[cfg(test)]
pub(crate) mod tests {
    use super::test_fixtures::*;
    use super::*;

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
    fn test_mask() -> Result<()> {
        let f = quickframe();
        // TODO document - keep if true or discard-if-true? At moment it's keep-if-true
        let mask = f.get::<IntT, _>().mask(|&v| v > 2);
        let f2 = f.filter_mask(&mask)?;
        assert_eq!(f2.get::<IntT, _>(), &[3, 4]);
        // Fails with incorrect len
        let mask2 = f2.get::<IntT, _>().mask(|&v| v > 2);
        assert!(f.filter_mask(&mask2).is_err());
        Ok(())
    }

    #[test]
    fn test_filter() {
        // basically same as above
        let f = quickframe();
        let f2 = f.filter::<IntT, _, _>(|&v| v > 2);
        assert_eq!(f2.get::<IntT, _>(), &[3, 4]);
    }

    #[test]
    fn test_groupby() -> Result<()> {
        // TODO special method for first column, or some kind of convenience builder
        let f: Frame3<IntT, FloatT, BoolT> = Frame::new()
            .addcol(vec![1, 3, 2, 3, 4, 2])?
            .addcol(vec![1., 2., 1., 1., 1., 1.])?
            .addcol(vec![true, false, true, false, true, false])?;
        define_col!(FloatSum, f64);
        define_col!(TrueCt, u32);
        // TODO can we rewrite to get rid of the dangling type parameters?
        let g = f
            .groupby::<IntT, _>()
            .acc::<FloatT, FloatSum, _, _>(|slice| slice.iter().map(|v| *v).sum())
            .acc::<BoolT, TrueCt, _, _>(|slice| slice.iter().map(|&&v| if v { 1 } else { 0 }).sum())
            .done();
        assert_eq!(g.get::<IntT, _>(), &[1, 3, 2, 4]);
        assert_eq!(g.get::<FloatSum, _>(), &[1., 3., 2., 1.]);
        assert_eq!(g.get::<TrueCt, _>(), &[1, 0, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_map_replace() {
        let f = quickframe();
        let f2 = f.map_replace_notnull::<FloatT, FloatT, _, _>(|&v| v * v);
        assert_eq!(f2.get::<FloatT, _>(), &[25., 16., 9., 4.]);
    }

    #[test]
    fn test_safely_drop() {
        use std::rc::Rc;
        use std::sync::Arc;
        define_col!(Arcs, Arc<u64>);
        define_col!(Rcs, Rc<u64>);
        // contains no nulls
        let _f0: Frame1<Arcs> = Frame::new()
            .addcol(Collection::from(vec![Arc::new(10), Arc::new(20)]))
            .unwrap();
        // contains nulls -> segfaults!
        let coll = Collection::new_opt(vec![Some(Rc::new(1)), None].into_iter());
        // let f1: Frame1<Arcs> = Frame::new()
        //     .addcol(Collection::new_opt(
        //         vec![None, None, None, Some(Arc::new(1))].into_iter(),
        //     )).unwrap();
        // ::std::mem::forget(f1);
        // let f2 = f1.clone();
    }

    #[test]
    fn test_inner_join() -> Result<()> {
        // TODO parse a text string once reading csvs is implemented
        let f1 = quickframe();
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Collection::new_opt(
                vec![Some(3), None, Some(2), Some(2)].into_iter(),
            ))?.addcol(Collection::new_opt(
                vec![None, Some(false), Some(true), Some(false)].into_iter(),
            ))?;
        let f3 = f1.inner_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(f3.get::<IntT, _>(), &[2, 2, 3]);
        assert_eq!(
            f3.get::<BoolT, _>(),
            &Collection::new_opt(vec![Some(true), Some(false), None].into_iter())
        );
        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(Collection::new_opt(
                vec![Some(3), None, Some(2), Some(2)].into_iter(),
            ))?.addcol(Collection::new_opt(
                vec![None, Some(5.), Some(4.), Some(3.)].into_iter(),
            ))?;

        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Collection::new_opt(
                vec![Some(2), Some(2), None, Some(1), Some(3)].into_iter(),
            ))?.addcol(Collection::new_opt(
                vec![None, Some(false), Some(true), Some(false), None].into_iter(),
            ))?;

        let f3 = f1.left_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(
            f3.get::<IntT, _>(),
            &Collection::new_opt(
                vec![Some(3), None, Some(2), Some(2), Some(2), Some(2)].into_iter()
            )
        );
        assert_eq!(
            f3.get::<BoolT, _>(),
            &Collection::new_opt(
                vec![None, None, None, Some(false), None, Some(false)].into_iter()
            )
        );
        Ok(())
    }

    #[test]
    fn test_outer_join_nones() -> Result<()> {
        let f1: Frame1<IntT> = Frame::new().addcol(Collection::new_opt(vec![None].into_iter()))?;
        let f2: Frame1<IntT> =
            Frame::new().addcol(Collection::new_opt(vec![None, None].into_iter()))?;
        let f3 = f1.outer_join(&f2);
        assert_eq!(
            f3.get(),
            &Collection::new_opt(vec![None, None, None].into_iter())
        );
        Ok(())
    }

    #[test]
    fn test_outer_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(Collection::new_opt(
                vec![Some(3), None, Some(2), None].into_iter(),
            ))?.addcol(Collection::new_opt(
                vec![Some(1.), Some(2.), None, Some(3.)].into_iter(),
            ))?;
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Collection::new_opt(
                vec![None, Some(3), Some(3), Some(2), Some(5)].into_iter(),
            ))?.addcol(Collection::new_opt(
                vec![Some(true), None, Some(false), Some(true), None].into_iter(),
            ))?;
        let f3 = f1.outer_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(
            f3.get::<IntT, _>(),
            &Collection::new_opt(
                vec![Some(3), Some(3), None, Some(2), None, None, Some(5)].into_iter()
            )
        );
        Ok(())
    }

    #[test]
    fn test_iter_rows() {
        let f: Frame3<IntT, FloatT, BoolT> = Frame::new()
            .addcol(vec![1, 2])
            .unwrap()
            .addcol(vec![5., 4.])
            .unwrap()
            .addcol(vec![false, true])
            .unwrap();
        let rows: Vec<(Option<&i64>, Option<&f64>, Option<&bool>)> = f.iter_rows().collect();
        assert_eq!(
            rows,
            vec![
                (Some(&1), Some(&5.), Some(&false)),
                (Some(&2), Some(&4.), Some(&true))
            ]
        );
    }
}
