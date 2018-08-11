use column::{Column, Mask};
pub use frame_alias::*;
use hlist::*;
use id;

use std::hash::Hash;

use Result;
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

    pub fn names(&self) -> Vec<&'static str>
    where
        H: HListExt,
    {
        self.hlist.get_names()
    }

    #[inline]
    pub fn get<Col, Index>(&self) -> &Column<Col::Output>
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
        Data: Into<Column<Col::Output>>,
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

    pub(crate) fn empty() -> Self
    where
        H: Insertable,
    {
        Frame {
            len: 0,
            hlist: H::empty(),
        }
    }

    pub(crate) unsafe fn insert_row(&mut self, product: H::Product)
    where
        H: Insertable,
    {
        self.hlist.insert(product);
        self.len += 1;
    }

    pub fn replace<Col: ColId, Index>(&mut self, newcol: Column<Col::Output>)
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
        Column<Col::Output>,
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
    H: HListClonable,
{
    pub fn inner_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
    ) -> Frame<<<Oth as Extractor<RCol, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList + Selector<RCol, RIx> + Extractor<RCol, RIx> + Concat<H> + HListClonable,
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
        Oth: HList + Selector<RCol, RIx> + Extractor<RCol, RIx> + Concat<H> + HListClonable,
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
        Oth: HList + Selector<RCol, RIx> + Extractor<RCol, RIx> + Concat<H> + HListClonable,
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
            Column::new(
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
            len: mask.true_count(),
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
    pub fn pop(self) -> (Column<Head::Output>, Frame<Tail>) {
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
    type ProductOptRef;
    fn get_product(&'a self, index: usize) -> Self::ProductOptRef;
}

impl<'a, Head, Tail> RowTuple<'a> for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: 'a,
    Tail: RowTuple<'a>,
{
    type ProductOptRef = Product<Option<&'a Head::Output>, Tail::ProductOptRef>;
    fn get_product(&'a self, index: usize) -> Self::ProductOptRef {
        Product(self.head.get(index).unwrap(), self.tail.get_product(index))
    }
}

impl<'a> RowTuple<'a> for HNil {
    type ProductOptRef = ();
    fn get_product(&'a self, _index: usize) -> Self::ProductOptRef {
        ()
    }
}

struct IterRows<'a, H: HList + 'a> {
    frame: &'a Frame<H>,
    index: usize,
}

impl<'a, H> Iterator for IterRows<'a, H>
where
    H: HList + RowTuple<'a>,
    <H as RowTuple<'a>>::ProductOptRef: Transformer,
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

// TODO this only works for single idents, ie "my string column" is not allowed
#[macro_export]
macro_rules! define_col {
    ($tyname:ident, $typ:ty) => {
        define_col!($tyname, $typ, $tyname);
    };
    ($tyname:ident, $typ:ty, $name:ident) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $tyname;
        impl ColId for $tyname {
            const NAME: &'static str = stringify!($name);
            type Output = $typ;
        }
    };
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
        assert_eq!(f2.len(), 2);
        assert_eq!(f2.get::<IntT, _>(), &[3, 4]);
        assert_eq!(f2.get::<FloatT, _>(), &[3., 2.]);
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
    fn test_inner_join() -> Result<()> {
        // TODO parse a text string once reading csvs is implemented
        let f1 = quickframe();
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Column::new(vec![Some(3), None, Some(2), Some(2)]))?
            .addcol(Column::new(vec![
                None,
                Some(false),
                Some(true),
                Some(false),
            ]))?;
        let f3 = f1.inner_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(f3.get::<IntT, _>(), &[2, 2, 3]);
        assert_eq!(
            f3.get::<BoolT, _>(),
            &Column::new(vec![Some(true), Some(false), None])
        );
        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(Column::new(vec![Some(3), None, Some(2), Some(2)]))?
            .addcol(Column::new(vec![None, Some(5.), Some(4.), Some(3.)]))?;

        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Column::new(vec![Some(2), Some(2), None, Some(1), Some(3)]))?
            .addcol(Column::new(vec![
                None,
                Some(false),
                Some(true),
                Some(false),
                None,
            ]))?;

        let f3 = f1.left_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(
            f3.get::<IntT, _>(),
            &Column::new(vec![Some(3), None, Some(2), Some(2), Some(2), Some(2)])
        );
        assert_eq!(
            f3.get::<BoolT, _>(),
            &Column::new(vec![None, None, None, Some(false), None, Some(false)])
        );
        Ok(())
    }

    #[test]
    fn test_outer_join_nones() -> Result<()> {
        let f1: Frame1<IntT> = Frame::new().addcol(Column::new(vec![None]))?;
        let f2: Frame1<IntT> = Frame::new().addcol(Column::new(vec![None, None]))?;
        let f3 = f1.outer_join(&f2);
        assert_eq!(f3.get(), &Column::new(vec![None, None, None]));
        Ok(())
    }

    #[test]
    fn test_outer_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .addcol(Column::new(vec![Some(3), None, Some(2), None]))?
            .addcol(Column::new(vec![Some(1.), Some(2.), None, Some(3.)]))?;
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .addcol(Column::new(vec![None, Some(3), Some(3), Some(2), Some(5)]))?
            .addcol(Column::new(vec![
                Some(true),
                None,
                Some(false),
                Some(true),
                None,
            ]))?;
        let f3 = f1.outer_join::<IntT, IntT, _, _, _>(&f2);
        assert_eq!(
            f3.get::<IntT, _>(),
            &Column::new(vec![Some(3), Some(3), None, Some(2), None, None, Some(5)])
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
