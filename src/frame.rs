use column::{ColId, Column, Mask, NamedColumn};
pub use frame_alias::*;
use hlist::*;
use {id, IndexVec};

use frunk::generic::Generic;
use frunk::hlist::{Plucker, Selector};

use std::hash::Hash;
use std::marker::PhantomData;

use Result;
// The HList implementation is a modified version of the one found in the `frunk` crate.
// See https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/
// for details. (In fact, that implementation is much more complete)

// TODO tag everything with #[must_use]

// ### Frame ###

#[derive(Debug, Clone, PartialEq)]
pub struct Frame<H: HList> {
    pub(crate) hlist: H,
    pub(crate) len: usize,
}

impl Frame<HNil> {
    pub fn new() -> Self {
        Frame {
            hlist: HNil,
            len: 0,
        }
    }
}

impl<T: ColId> Frame<HCons<T, HNil>> {
    pub fn with<I: Into<NamedColumn<T>>>(col: I) -> Self {
        let col = col.into();
        Frame {
            len: col.len(),
            hlist: HCons {
                head: col,
                tail: HNil,
            },
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
        self.hlist.len()
    }

    pub fn names(&self) -> Vec<&'static str>
    where
        H: HListExt,
    {
        self.hlist.get_names()
    }

    #[inline]
    pub fn get<Col, Index>(&self) -> &NamedColumn<Col>
    where
        Col: ColId,
        H: Selector<NamedColumn<Col>, Index>,
    {
        Selector::get(&self.hlist)
    }

    // TODO: alternative would be to explicitly pass the Col token
    // TODO: what would be nicer is a `setcol` func which either adds or modifies
    pub fn addcol<Col, Data>(self, coll: Data) -> Result<Frame<HCons<Col, H>>>
    where
        Col: ColId,
        Data: Into<NamedColumn<Col>>,
    {
        let coll = coll.into();
        if self.hlist.len() != 0 && coll.len() != self.len {
            bail!("Mismatched lengths ({} and {})", self.len(), coll.len())
        } else {
            Ok(Frame {
                len: coll.len(),
                hlist: self.hlist.prepend(coll),
            })
        }
    }

    pub fn replace<Col: ColId, Index>(&mut self, newcol: NamedColumn<Col>)
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
        NamedColumn<Col>,
        Frame<<H as Plucker<NamedColumn<Col>, Index>>::Remainder>,
    )
    where
        Col: ColId,
        H: Plucker<NamedColumn<Col>, Index>,
        <H as Plucker<NamedColumn<Col>, Index>>::Remainder: HList,
    {
        let (v, hlist) = self.hlist.pluck();
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
        let len = if self.hlist.len() == 0 {
            other.len()
        } else if other.hlist.len() == 0 {
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
    H: HListClonable,
{
    pub fn inner_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + Concat<H>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Hash,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx>,
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
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + Concat<H>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Hash,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx>,
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
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + Concat<H>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Hash,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx> + Replacer<LCol, LIx>,
    {
        let left = self.get::<LCol, _>();
        let right = other.get::<RCol, _>();
        let (leftixs, rightixs) = left.outer_join_locs(right);
        let mut leftframe = self.copy_locs_opt(&leftixs);
        let rightframe = other.copy_locs_opt(&rightixs);
        let (rjoined, rightframe) = rightframe.extract::<RCol, _>();
        let joined = {
            let ljoined = leftframe.get::<LCol, _>();
            NamedColumn::from(Column::new(
                ljoined
                    .iter()
                    .zip(rjoined.iter())
                    .map(|(left, right)| left.or(right).cloned()),
            ))
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
        H: Selector<NamedColumn<Col>, Index>,
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
            len: mask.true_count(),
        })
    }

    pub fn groupby<Col, Index>(self) -> GroupBy<H, HCons<Col, HNil>>
    where
        Col: ColId,
        Col::Output: Eq + Clone + Hash,
        H: Selector<NamedColumn<Col>, Index>,
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

    // pub fn reshape<Shaped>(&self) -> Frame<Shaped>
    //     where Shaped:
    // {

    // }
}

impl<Col: ColId, Tail: HList> Frame<HCons<Col, Tail>> {
    #[inline(always)]
    pub fn pop(self) -> (NamedColumn<Col>, Frame<Tail>) {
        let tail = Frame {
            hlist: self.hlist.tail,
            len: self.len,
        };
        (self.hlist.head, tail)
    }
}

impl<'a, H> Frame<H>
where
    H: HList + RowTuple<'a>,
    <H as RowTuple<'a>>::ProductOptRef: Transformer,
{
    pub fn get_row(
        &'a self,
        index: usize,
    ) -> Option<<<H as RowTuple>::ProductOptRef as Transformer>::Flattened> {
        if index >= self.len() {
            return None;
        }
        let nested = self.hlist.get_product(index);
        Some(H::ProductOptRef::flatten(nested))
    }

    pub fn iter_rows(
        &'a self,
    ) -> impl Iterator<Item = <<H as RowTuple>::ProductOptRef as Transformer>::Flattened> {
        IterRows {
            frame: &self,
            index: 0,
        }
    }

    // pub fn iter_elems<T>(&'a self) -> impl Iterator<Item = T>
    // where
    //     <H::ProductOptRef as Transformer>::Flattened: Generic,
    //     T: Generic<Repr = <<H::ProductOptRef as Transformer>::Flattened as Generic>::Repr>,
    // {
    //     IterElems {
    //         frame: &self,
    //         index: 0,
    //         ret: PhantomData,
    //     }
    // }
}

// ### RowTuple ###

pub trait RowTuple<'a> {
    type ProductOptRef;
    fn get_product(&'a self, index: usize) -> Self::ProductOptRef;
}

impl<'a, Head, Tail> RowTuple<'a> for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: 'a,
    Tail: RowTuple<'a>,
{
    type ProductOptRef = HConsFrunk<Option<&'a Head::Output>, Tail::ProductOptRef>;
    fn get_product(&'a self, index: usize) -> Self::ProductOptRef {
        HConsFrunk {
            head: self.head.get(index).unwrap(),
            tail: self.tail.get_product(index),
        }
    }
}

impl<'a> RowTuple<'a> for HNil {
    type ProductOptRef = HNil;
    fn get_product(&'a self, _index: usize) -> Self::ProductOptRef {
        HNil
    }
}

struct IterRows<'a, H: HList + 'a> {
    frame: &'a Frame<H>,
    index: usize,
}

impl<'a, H> Iterator for IterRows<'a, H>
where
    H: HList + RowTuple<'a>,
    H::ProductOptRef: Transformer,
{
    type Item = <<H as RowTuple<'a>>::ProductOptRef as Transformer>::Flattened;
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

// TODO this needs more thought. Need to grab the relevent columns,
// make a new HList from said columns, iterate over them, clone out the
// values into a tuple and convert to struct

// We could get rid of the RowTuples stuff as frunk already does it. Iterating
// as a tuple could just be a particular instance of iterator over the generic
struct IterElems<'a, H: HList + 'a, T> {
    frame: &'a Frame<H>,
    index: usize,
    ret: PhantomData<T>,
}

impl<'a, H, T> Iterator for IterElems<'a, H, T>
where
    H: HList + RowTuple<'a>,
    H::ProductOptRef: Transformer,
    <H::ProductOptRef as Transformer>::Flattened: Generic,
    T: Generic<Repr = <<H::ProductOptRef as Transformer>::Flattened as Generic>::Repr>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        // match (*self.frame).get_row(self.index) {
        //     Some(r) => {
        //         self.index += 1;
        //         Some(r)
        //     }
        //     None => None,
        // }
        unimplemented!()
    }
}

// ### GroupBy ###

pub struct GroupBy<H: HList, G: HList> {
    frame: Frame<H>,
    grouping_index: Vec<IndexVec>,
    grouped_frame: Frame<G>,
}

impl<H, G> GroupBy<H, G>
where
    H: HList,
    G: HList,
{
    // TODO Accumulate WITHOUT nulls
    // Also need an acc WITH nulls
    // Also this is very inefficient and uses iterate_indices which is also inefficient
    pub fn accumulate<Col, NewCol, Index, AccFn>(self, func: AccFn) -> GroupBy<H, HCons<NewCol, G>>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<NamedColumn<Col>, Index>,
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
        let grouped_frame = self.grouped_frame.addcol(NamedColumn::with(res)).unwrap();
        GroupBy {
            frame: self.frame,
            grouping_index: self.grouping_index,
            grouped_frame,
        }
    }

    /// Shorthand for accumulate
    pub fn acc<Col, NewCol, Index, AccFn>(self, func: AccFn) -> GroupBy<H, HCons<NewCol, G>>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<NamedColumn<Col>, Index>,
        AccFn: Fn(&[&Col::Output]) -> NewCol::Output,
    {
        self.accumulate(func)
    }

    pub fn done(self) -> Frame<G> {
        self.grouped_frame
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    define_col!(IntT, i64, int_col);
    define_col!(StringT, String, string_col);
    define_col!(FloatT, f64, float_col);
    define_col!(BoolT, bool, bool_col);

    pub(crate) type Data3 = Frame3<IntT, FloatT, StringT>;

    pub(crate) fn quickframe() -> Data3 {
        Frame::with(col![1, 2, None, 3, 4])
            .addcol(col![5., None, 3., 2., 1.])
            .unwrap()
            .addcol(
                r#"this,'" is the words here"#
                    .split(' ')
                    .map(String::from)
                    .map(Some),
            ).unwrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::test_fixtures::*;
    use super::*;

    // TODO convenient column literal macro
    #[test]
    fn test_basic_frame() -> Result<()> {
        let f: Data3 = Frame::with(col![10i64])
            .addcol(col![1.23f64])?
            // TODO String doesn't work with col! macro
            .addcol(vec![Some("Hello".into())])?;
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
        let f: Frame2<IntT, IntT> = Frame::with(col![10]).addcol(col![1])?;
        let (v, _) = f.extract::<IntT, First>();
        assert_eq!(v, &[10]);
        Ok(())
    }

    #[test]
    fn test_add_coldef() -> Result<()> {
        let f: Data3 = Frame::new()
            .addcol(col![10i64])?
            .addcol(col![1.23f64])?
            .addcol(vec![Some(String::from("Hello"))])?;
        define_col!(Added, i64);
        let f = f.addcol::<Added, _>(col![123])?;
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
                Frame::new().addcol(col![10i64])?.addcol(col![1.23f64])?;
            let f2: Frame1<StringT> = Frame::new().addcol(vec![Some(String::from("Hello"))])?;
            let _f3: Frame3<IntT, FloatT, StringT> = f1.concat(f2)?;
        }

        {
            let f1 = Frame::new();
            let f2 = Frame::new().addcol::<IntT, _>(col![1, 2, 3])?;
            let f3 = Frame::new().addcol::<FloatT, _>(col![1., 2.])?;
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
        assert_eq!(f2.len(), 2);
        assert_eq!(f2.get::<IntT, _>(), &[3, 4]);
        assert_eq!(f2.get::<FloatT, _>(), &[2., 1.]);
    }

    #[test]
    fn test_groupby() -> Result<()> {
        // TODO special method for first column, or some kind of convenience builder
        let f: Frame3<IntT, FloatT, BoolT> = Frame::new()
            .addcol(col![1, 3, 2, 3, 4, 2])?
            .addcol(col![1., 2., 1., 1., 1., 1.])?
            .addcol(col![true, false, true, false, true, false])?;
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
        assert_eq!(f2.get::<FloatT, _>(), &col![25., None, 9., 4., 1.]);
    }

    #[test]
    fn test_inner_join() -> Result<()> {
        // TODO parse a text string once reading csvs is implemented
        let f1 = quickframe();
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(col![3, None, 2, 2])?
            .addcol(col![None, false, true, false])?;
        let f3 = f1.inner_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(f3.get::<IntT, _>(), &[2, 2, 3]);
        assert_eq!(f3.get::<BoolT, _>(), &col![true, false, None]);
        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(col![3, None, 2, 2])?
            .addcol(col![None, 5., 4., 3.])?;

        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(col![2, 2, None, 1, 3])?
            .addcol(col![None, false, true, false, None])?;

        let f3 = f1.left_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(f3.get::<IntT, _>(), &col![3, None, 2, 2, 2, 2]);
        assert_eq!(
            f3.get::<BoolT, _>(),
            &col![None, None, None, false, None, false]
        );
        Ok(())
    }

    #[test]
    fn test_outer_join_nones() -> Result<()> {
        let f1: Frame1<IntT> = Frame::new().addcol(col![None])?;
        let f2: Frame1<IntT> = Frame::new().addcol(col![None, None])?;
        let f3 = f1.outer_join(&f2);
        assert_eq!(f3.get(), &col![None, None, None]);
        Ok(())
    }

    #[test]
    fn test_outer_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(col![3, None, 2, None])?
            .addcol(col![1., 2., None, 3.])?;
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(col![None, 3, 3, 2, 5])?
            .addcol(col![true, None, false, true, None])?;
        let f3 = f1.outer_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(f3.get::<IntT, _>(), &col![3, 3, None, 2, None, None, 5]);
        Ok(())
    }

    #[test]
    fn test_iter_rows() {
        let f: Frame3<IntT, FloatT, BoolT> = Frame::new()
            .addcol(col![1, 2, None])
            .unwrap()
            .addcol(col![None, 5., 4.])
            .unwrap()
            .addcol(col![false, None, true])
            .unwrap();
        let rows: Vec<(Option<&i64>, Option<&f64>, Option<&bool>)> = f.iter_rows().collect();
        assert_eq!(
            rows,
            vec![
                (Some(&1), None, Some(&false)),
                (Some(&2), Some(&5.), None),
                (None, Some(&4.), Some(&true))
            ],
        );
    }

    // #[test]
    // fn test_ad_hoc_iter() {
    //     let f = quickframe();
    //     let iter = f.iter_rows::<(IntT, FloatT, IntT)>();
    //     assert_eq!(iter.next(), Some((Some(&1), Some(&5.), Some("this,'\"".to_string()))));
    // }
}
