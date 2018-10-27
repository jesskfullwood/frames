pub use frunk::hlist::{HList, HNil, Plucker, Selector};

use column::{ColId, Column, IndexVec, Mask, NamedColumn};
pub use frame_typedef::*;
use hlist::{
    Appender, Concat, HCons, HListClonable, HListExt, Mapper, Replacer, RowHList, Stringify,
    Transformer,
};
use id;
use io;

use std::fmt;
use std::marker::PhantomData;

use Result;

// TODO tag everything with #[must_use]

// ### Frame ###

#[derive(Debug, Clone, PartialEq)]
pub struct Frame<H: HList> {
    pub(crate) hlist: H,
    pub(crate) len: usize,
}

impl Frame<HNil> {
    pub fn new() -> Frame0 {
        Frame {
            hlist: HNil,
            len: 0,
        }
    }

    pub fn with<Col, Data>(data: Data) -> Frame1<Col>
    where
        Col: ColId,
        Data: Into<NamedColumn<Col>>,
    {
        let col = data.into();
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

#[macro_export]
macro_rules! frame {
    ($($col:expr),* $(,)*) => {
        {
            let f = $crate::Frame::new();
            $(
                let f = f.add($col).unwrap();
            )*;
            f
        }
    }
}

impl<H> fmt::Display for Frame<H>
where
    H: HList + for<'a> RowHList<'a>,
    for<'a> <H as RowHList<'a>>::ProductOptRef: Stringify,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO I think this doesn't work because of this bug:
        // https://github.com/rust-lang/rust/issues/50346
        for row in self.iter_rows_hlist() {
            let strs = row.stringify();
            for s in strs {
                write!(f, "{}", s)?
            }
        }
        Ok(())
    }
}

impl<H: HList> Frame<H> {
    /// Parse a CSV into a frame
    pub fn from_csv(path: impl AsRef<::std::path::Path>) -> Result<Frame<H>>
    where
        H: HList + io::Insertable,
        H::Product: Transformer,
        <H::Product as Transformer>::Flattened: ::serde::de::DeserializeOwned,
    {
        io::read_csv(path)
    }

    /// Number of rows in the frame
    pub fn nrows(&self) -> usize {
        self.len
    }

    /// Number of columns in the frame
    pub fn ncols(&self) -> usize {
        self.hlist.len()
    }

    /// Fetch vec of column names
    pub fn colnames(&self) -> Vec<&'static str>
    where
        H: HListExt,
    {
        let mut names = Vec::with_capacity(self.ncols());
        self.hlist.get_names(&mut names);
        names
    }

    #[allow(unused_variables)]
    /// Get column from specified column ident
    pub fn get<Col, Index>(&self, col: Col) -> &NamedColumn<Col>
    where
        Col: ColId,
        H: Selector<NamedColumn<Col>, Index>,
    {
        Selector::get(&self.hlist)
    }

    #[allow(unused_variables)]
    /// Mutably column from specified column ident
    fn get_mut<Col, Index>(&mut self, col: Col) -> &mut NamedColumn<Col>
    where
        Col: ColId,
        H: Selector<NamedColumn<Col>, Index>,
    {
        // NOTE This function is potentialy unsafe, if swap is called on
        // the mut ref to insert a col with the wrong length
        Selector::get_mut(&mut self.hlist)
    }

    /// Get column from specifying column ident at the type level.
    ///
    /// This feels a little less natural to type than `get` but is more convenient
    /// in circumstances where an ident type is not at hand
    pub fn have<Col, Index>(&self) -> &NamedColumn<Col>
    where
        Col: ColId,
        H: Selector<NamedColumn<Col>, Index>,
    {
        Selector::get(&self.hlist)
    }

    // TODO: what would be nicer is a `setcol` func which either adds or modifies
    /// Add a column to the end of the frame
    /// The column must have the same number of rows as the existing frame (excepting empty frames).
    pub fn add<Col, Data>(self, col: Data) -> Result<Frame<H::FromRoot>>
    where
        H: Appender<NamedColumn<Col>>,
        H::FromRoot: HList,
        Col: ColId,
        Data: Into<NamedColumn<Col>>,
    {
        let col = col.into();
        if self.hlist.len() != 0 && col.len() != self.nrows() {
            bail!(
                "Bad column length (expected {}, found {})",
                self.nrows(),
                col.len()
            )
        } else {
            Ok(Frame {
                len: col.len(),
                hlist: self.hlist.append(col),
            })
        }
    }

    // TODO is this function useful?
    /// Replace an existing column with a new column of the same type.
    /// This operation does not change the type of the frame.
    ///
    /// # Panics
    ///
    /// panics if the replacement column has a different number of rows from the existing frame
    pub fn replace<Col: ColId, Index>(&mut self, new_col: NamedColumn<Col>)
    where
        H: Replacer<Col, Index>,
    {
        if new_col.len() != self.nrows() {
            panic!(
                "Bad columns length (expected {}, found {}",
                self.nrows(),
                new_col.len()
            )
        }
        self.hlist.replace(new_col)
    }

    pub fn map_replace_all<Col, NewCol, Index, F>(
        self,
        _col: Col,
        _newcol: NewCol,
        func: F,
    ) -> Frame<<H as Mapper<Col, NewCol, Index>>::Mapped>
    where
        Col: ColId,
        NewCol: ColId,
        H: Mapper<Col, NewCol, Index>,
        F: Fn(Option<&Col::Output>) -> Option<NewCol::Output>,
    {
        Frame {
            hlist: self.hlist.map_replace_all(func),
            len: self.len,
        }
    }

    pub fn map_replace<Col, NewCol, Index, F>(
        self,
        _col: Col,
        _newcol: NewCol,
        func: F,
    ) -> Frame<<H as Mapper<Col, NewCol, Index>>::Mapped>
    where
        Col: ColId,
        NewCol: ColId,
        H: Mapper<Col, NewCol, Index>,
        F: Fn(&Col::Output) -> NewCol::Output,
    {
        Frame {
            hlist: self.hlist.map_replace(func),
            len: self.len,
        }
    }

    pub fn map_insert<Col, NewCol, Index, F>(
        self,
        col_name: Col,
        newcol_name: NewCol,
        func: F,
    ) -> Frame<H::FromRoot>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<NamedColumn<Col>, Index>,
        H: Appender<NamedColumn<NewCol>>,
        H::FromRoot: HList,
        F: Fn(&Col::Output) -> NewCol::Output,
    {
        let newcol: NamedColumn<NewCol> = {
            let col = self.get(col_name);
            NamedColumn::new(col.map(func))
        };
        self.add(newcol).unwrap() // safe to unwrap as we know it is the same length
    }

    pub fn map_in_place<Col, Index, F>(&mut self, col: Col, f: F)
    where
        H: Selector<NamedColumn<Col>, Index>,
        Col: ColId,
        F: Fn(&mut Col::Output),
    {
        let col = self.get_mut(col);
        col.iter_mut().for_each(|v| f(v))
    }

    #[inline(always)]
    #[allow(unused_variables)]
    pub fn extract<Col, Index>(
        self,
        col: Col,
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

    pub fn concat_front<C: HList + Concat<H>>(self, other: Frame<C>) -> Result<Frame<C::Combined>> {
        other.concat(self)
    }

    pub fn concat<C: HList>(self, other: Frame<C>) -> Result<Frame<H::Combined>>
    where
        H: Concat<C>,
    {
        let len = if self.hlist.len() == 0 {
            other.nrows()
        } else if other.hlist.len() == 0 {
            self.nrows()
        } else if self.nrows() != other.nrows() {
            bail!(
                "Mismatched lengths ({} and {})",
                other.nrows(),
                self.nrows()
            )
        } else {
            self.nrows()
        };
        Ok(Frame {
            len,
            hlist: self.hlist.concat(other.hlist),
        })
    }
}

impl<H> Frame<H>
where
    H: HListClonable,
{
    pub fn inner_join<LCol, RCol, Oth, LIx, RIx>(
        &self,
        other: &Frame<Oth>,
        lcol: LCol,
        rcol: RCol,
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Ord,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx>,
    {
        let left = self.get(lcol);
        let right = other.get(rcol);
        let (leftixs, rightixs) = left.inner_join_locs(right);
        let leftframe = self.copy_locs(&leftixs);
        let rightframe = other.copy_locs(&rightixs);
        let (_, rightframe) = rightframe.extract(rcol);
        leftframe.concat_front(rightframe).unwrap()
    }

    pub fn left_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
        lcol: LCol,
        rcol: RCol,
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + Concat<H>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Ord,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx>,
    {
        let left = self.get(lcol);
        let right = other.get(rcol);
        let (leftixs, rightixs) = left.left_join_locs(right);
        let leftframe = self.copy_locs(&leftixs);
        let rightframe = other.copy_locs_opt(&rightixs);
        let (_, rightframe) = rightframe.extract(rcol);
        leftframe.concat_front(rightframe).unwrap()
    }

    // TODO are all these bounds necessary?
    pub fn outer_join<LCol, RCol, Oth, LIx, RIx>(
        self,
        other: &Frame<Oth>,
        lcol: LCol,
        rcol: RCol,
    ) -> Frame<<<Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder as Concat<H>>::Combined>
    where
        Oth: HList
            + Selector<NamedColumn<RCol>, RIx>
            + Plucker<NamedColumn<RCol>, RIx>
            + Concat<H>
            + HListClonable,
        <Oth as Plucker<NamedColumn<RCol>, RIx>>::Remainder: Concat<H> + HList,
        LCol: ColId,
        LCol::Output: Eq + Clone + Ord,
        RCol: ColId<Output = LCol::Output>,
        H: Selector<NamedColumn<LCol>, LIx> + Replacer<LCol, LIx>,
    {
        let left = self.get(lcol);
        let right = other.get(rcol);
        let (leftixs, rightixs) = left.outer_join_locs(right);
        let mut leftframe = self.copy_locs_opt(&leftixs);
        let rightframe = other.copy_locs_opt(&rightixs);
        let (rjoined, rightframe) = rightframe.extract(rcol);
        let joined = {
            let ljoined = leftframe.get(lcol);
            NamedColumn::from(Column::new_null(
                ljoined
                    .iter_null()
                    .zip(rjoined.iter_null())
                    .map(|(left, right)| left.or(right).cloned()),
            ))
        };
        leftframe.replace(joined);
        leftframe.concat_front(rightframe).unwrap()
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

    pub fn filter<Col, Index, F>(&self, col: Col, func: F) -> Self
    where
        Col: ColId,
        F: Fn(&Col::Output) -> bool,
        H: Selector<NamedColumn<Col>, Index>,
    {
        // TODO also add filter2, filter3...
        let mask = self.get(col).mask(func);
        // We know the mask is the right length
        self.filter_mask(&mask).unwrap()
    }

    pub fn filter_mask(&self, mask: &Mask) -> Result<Self> {
        if mask.len() != self.nrows() {
            bail!("Mismatched lengths ({} and {})", self.nrows(), mask.len())
        }
        Ok(Frame {
            hlist: self.hlist.filter_mask(mask),
            len: mask.true_count(),
        })
    }

    pub fn groupby<Col, Index>(self, col: Col) -> GroupBy<H, HCons<Col, HNil>>
    where
        Col: ColId,
        Col::Output: Eq + Clone + Ord,
        H: Selector<NamedColumn<Col>, Index>,
    {
        let (grouping_index, grouped_col) = {
            // lifetimes workaround
            let grouping_col = self.get(col);
            let mut index = grouping_col.index_values();
            index.sort_unstable();
            let groupedcol = grouping_col.copy_first_locs(&index);
            (index, groupedcol)
        };
        let grouped_frame = Frame::new().add(grouped_col).unwrap();
        GroupBy {
            frame: self,
            grouping_index,
            grouped_frame,
        }
    }

    // TODO reshape
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
    H: HList + RowHList<'a>,
{
    pub fn get_row_hlist(&'a self, index: usize) -> Option<<H as RowHList>::ProductOptRef> {
        // NOTE: This check is done here so it doesn't have to be done by the inner HList
        if index >= self.nrows() {
            return None;
        }
        Some(self.hlist.get_row(index))
    }

    fn iter_rows_hlist(&'a self) -> impl Iterator<Item = <H as RowHList>::ProductOptRef> {
        IterRows {
            frame: &self,
            index: 0,
            tag: PhantomData,
        }
    }
}

impl<'a, H> Frame<H>
where
    H: HList + RowHList<'a>,
    <H as RowHList<'a>>::ProductOptRef: Transformer,
{
    pub fn get_row(
        &'a self,
        index: usize,
    ) -> Option<<<H as RowHList>::ProductOptRef as Transformer>::Flattened> {
        self.get_row_hlist(index).map(H::ProductOptRef::flatten)
    }

    pub fn iter_rows(
        &'a self,
    ) -> impl Iterator<Item = <<H as RowHList>::ProductOptRef as Transformer>::Flattened> {
        let tag: PhantomData<IterTuple> = PhantomData;
        IterRows {
            frame: &self,
            index: 0,
            tag,
        }
    }

    // TODO
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

struct IterRows<'a, H: HList + 'a, T> {
    frame: &'a Frame<H>,
    index: usize,
    tag: PhantomData<T>,
}

enum IterHList {}
enum IterTuple {}

impl<'a, H> Iterator for IterRows<'a, H, IterHList>
where
    H: HList + RowHList<'a>,
{
    type Item = <H as RowHList<'a>>::ProductOptRef;
    fn next(&mut self) -> Option<Self::Item> {
        match (*self.frame).get_row_hlist(self.index) {
            Some(r) => {
                self.index += 1;
                Some(r)
            }
            None => None,
        }
    }
}

impl<'a, H> Iterator for IterRows<'a, H, IterTuple>
where
    H: HList + RowHList<'a>,
    H::ProductOptRef: Transformer,
{
    type Item = <<H as RowHList<'a>>::ProductOptRef as Transformer>::Flattened;
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

// We could get rid of the RowHList stuff as frunk already does it. Iterating
// as a tuple could just be a particular instance of iterator over the generic
// struct IterElems<'a, H: HList + 'a, T> {
//     frame: &'a Frame<H>,
//     index: usize,
//     ret: PhantomData<T>,
// }

// TODO iter elems as struct, somehow
// impl<'a, H, T> Iterator for IterElems<'a, H, T>
// where
//     H: HList + RowHList<'a>,
//     H::ProductOptRef: Transformer,
//     <H::ProductOptRef as Transformer>::Flattened: Generic,
//     T: Generic<Repr = <<H::ProductOptRef as Transformer>::Flattened as Generic>::Repr>,
// {
//     type Item = T;
//     fn next(&mut self) -> Option<Self::Item> {
//         // match (*self.frame).get_row(self.index) {
//         //     Some(r) => {
//         //         self.index += 1;
//         //         Some(r)
//         //     }
//         //     None => None,
//         // }
//         unimplemented!()
//     }
// }

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
    pub fn accumulate<Col, NewCol, Index, AccFn>(
        self,
        col: Col,
        _newcol: NewCol,
        func: AccFn,
    ) -> GroupBy<H, G::FromRoot>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<NamedColumn<Col>, Index>,
        G: Appender<NamedColumn<NewCol>>,
        G::FromRoot: HList,
        AccFn: Fn(&[&Col::Output]) -> NewCol::Output,
    {
        let res: Vec<NewCol::Output> = {
            let grouped_col = self.frame.get(col);
            self.grouping_index
                .iter()
                .map(|grp| {
                    // TODO could this be done with an iterator instead of allocating a vec?
                    let to_acc: Vec<&Col::Output> = grouped_col
                        .iterate_indices(grp.iter().cloned())
                        .filter_map(id)
                        .collect();
                    func(&to_acc)
                })
                .collect()
        };
        let grouped_frame = self.grouped_frame.add(NamedColumn::with(res)).unwrap();
        GroupBy {
            frame: self.frame,
            grouping_index: self.grouping_index,
            grouped_frame,
        }
    }

    /// Shorthand for accumulate
    pub fn acc<Col, NewCol, Index, AccFn>(
        self,
        col: Col,
        newcol: NewCol,
        func: AccFn,
    ) -> GroupBy<H, G::FromRoot>
    where
        Col: ColId,
        NewCol: ColId,
        H: Selector<NamedColumn<Col>, Index>,
        G: Appender<NamedColumn<NewCol>>,
        G::FromRoot: HList,
        AccFn: Fn(&[&Col::Output]) -> NewCol::Output,
    {
        self.accumulate(col, newcol, func)
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
        Frame::with(col![1, 2, NA, 3, 4])
            .add(col![5., NA, 3., 2., 1.])
            .unwrap()
            .add(r#"this,'" is the words here"#.split(' ').map(String::from).map(Some))
            .unwrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::test_fixtures::*;
    use super::*;
    use frunk::indices::{Here, There};

    #[test]
    fn test_basic_frame() -> Result<()> {
        let f: Data3 = Frame::with(col![10i64])
            .add(col![1.23f64])?
            // TODO String doesn't work with col! macro
            .add(vec![Some("Hello".into())])?;
        assert_eq!(f.ncols(), 3);
        assert_eq!(f.len, 1);
        {
            let f = f.get(FloatT);
            assert_eq!(f, &[1.23f64])
        }
        let (v, f) = f.extract(IntT);
        assert_eq!(v, &[10]);
        let (v, f) = f.pop();
        assert_eq!(v, &[1.23]);
        let (v, _) = f.extract(StringT);
        assert_eq!(v, &[String::from("Hello")]);
        Ok(())
    }

    #[test]
    fn test_double_insert() -> Result<()> {
        let f: Frame2<IntT, IntT> = Frame::with(col![10]).add(col![101])?;
        let (v, _) = f.extract::<IntT, There<Here>>(IntT);
        assert_eq!(v, &[101]);
        Ok(())
    }

    #[test]
    fn test_add_coldef() -> Result<()> {
        let f: Data3 = Frame::new()
            .add(col![10i64])?
            .add(col![1.23f64])?
            .add(vec![Some(String::from("Hello"))])?;
        define_col!(Added, i64);
        let f = f.add::<Added, _>(col![123])?;
        let v = f.get(Added);
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
            let f1: Frame2<IntT, FloatT> = Frame::new().add(col![10i64])?.add(col![1.23f64])?;
            let f2: Frame1<StringT> = Frame::new().add(vec![Some(String::from("Hello"))])?;
            let _f3: Frame3<IntT, FloatT, StringT> = f1.concat(f2)?;
        }

        {
            let f1 = Frame::new();
            let f2 = Frame::new().add::<IntT, _>(col![1, 2, 3])?;
            let f3 = Frame::new().add::<FloatT, _>(col![1., 2.])?;
            let f4: Frame1<_> = f1.concat(f2)?;
            assert!(f4.concat(f3).is_err()) // mismatched types
        }

        Ok(())
    }

    #[test]
    fn test_mask() -> Result<()> {
        let f = quickframe();
        // TODO document - keep if true or discard-if-true? At moment it's keep-if-true
        let mask = f.get(IntT).mask(|&v| v > 2);
        let f2 = f.filter_mask(&mask)?;
        assert_eq!(f2.get(IntT), &[3, 4]);
        // Fails with incorrect len
        let mask2 = f2.get(IntT).mask(|&v| v > 2);
        assert!(f.filter_mask(&mask2).is_err());
        Ok(())
    }

    #[test]
    fn test_filter() {
        // basically same as above
        let f = quickframe();
        let f2 = f.filter(IntT, |&v| v > 2);
        assert_eq!(f2.nrows(), 2);
        assert_eq!(f2.get(IntT), &[3, 4]);
        assert_eq!(f2.get(FloatT), &[2., 1.]);
    }

    #[test]
    fn test_groupby() -> Result<()> {
        let f: Frame3<IntT, FloatT, BoolT> = Frame::with(col![1, 3, 2, 3, 4, 2])
            .add(col![1., 2., 1., 1., 1., 1.])?
            .add(col![true, false, true, false, true, false])?;
        define_col!(FloatSum, f64);
        define_col!(TrueCt, u32);
        let g = f
            .groupby(IntT)
            .acc(FloatT, FloatSum, |slice| slice.iter().map(|v| *v).sum())
            .acc(BoolT, TrueCt, |slice| {
                slice.iter().map(|&&v| if v { 1 } else { 0 }).sum()
            })
            .done();
        assert_eq!(g.get(IntT), &[1, 3, 2, 4]);
        assert_eq!(g.get(FloatSum), &[1., 3., 2., 1.]);
        assert_eq!(g.get(TrueCt), &[1, 0, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_map_replace() {
        let f = quickframe();
        let f2 = f.map_replace(FloatT, FloatT, |&v| v * v);
        assert_eq!(f2.get(FloatT), &col![25., NA, 9., 4., 1.]);
    }

    #[test]
    fn test_inner_join() -> Result<()> {
        // TODO parse a text string once reading csvs is implemented
        let f1 = quickframe();
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .add(col![3, NA, 2, 2])?
            .add(col![NA, false, true, false])?;
        let f3 = f1.inner_join(&f2, IntT, IntT);
        assert_eq!(f3.get(IntT), &[2, 2, 3]);
        assert_eq!(f3.get(BoolT), &col![true, false, NA]);
        Ok(())
    }

    #[test]
    fn test_inner_join_with_duplicates() {
        let f1 = Frame::with((0..50_000).map(|v| v / 2).map(Some));
        let joined = f1.inner_join(&f1, IntT, IntT);
        assert_eq!(joined.nrows(), 100_000);
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .add(col![3, NA, 2, 2])?
            .add(col![NA, 5., 4., 3.])?;

        let f2: Frame2<IntT, BoolT> = Frame::new()
            .add(col![2, 2, NA, 1, 3])?
            .add(col![NA, false, true, false, NA])?;

        let f3 = f1.left_join(&f2, IntT, IntT);
        assert_eq!(f3.get(IntT), &col![3, NA, 2, 2, 2, 2]);
        assert_eq!(f3.get(BoolT), &col![NA, NA, NA, false, NA, false]);
        Ok(())
    }

    #[test]
    fn test_outer_join_nones() -> Result<()> {
        let f1: Frame1<IntT> = Frame::new().add(col![NA])?;
        let f2: Frame1<IntT> = Frame::new().add(col![NA, NA])?;
        let f3 = f1.outer_join(&f2, IntT, IntT);
        assert_eq!(f3.get(IntT), &col![NA, NA, NA]);
        Ok(())
    }

    #[test]
    fn test_outer_join() -> Result<()> {
        let f1: Frame2<IntT, FloatT> = Frame::new()
            .add(col![3, NA, 2, NA])?
            .add(col![1., 2., NA, 3.])?;
        let f2: Frame2<IntT, BoolT> = Frame::new()
            .add(col![NA, 3, 3, 2, 5])?
            .add(col![true, NA, false, true, NA])?;
        let f3 = f1.outer_join(&f2, IntT, IntT);
        assert_eq!(f3.get(IntT), &col![3, 3, NA, 2, NA, NA, 5]);
        Ok(())
    }

    #[test]
    fn test_iter_rows() {
        let f: Frame3<IntT, FloatT, BoolT> = Frame::new()
            .add(col![1, 2, NA])
            .unwrap()
            .add(col![NA, 5., 4.])
            .unwrap()
            .add(col![false, NA, true])
            .unwrap();
        let rows: Vec<(Option<&i64>, Option<&f64>, Option<&bool>)> = f.iter_rows().collect();
        assert_eq!(
            rows,
            vec![
                (Some(&1), None, Some(&false)),
                (Some(&2), Some(&5.), None),
                (None, Some(&4.), Some(&true)),
            ],
        );
    }

    #[test]
    fn test_frame_macro() {
        let _empty = frame![];
        let f: Frame2<IntT, FloatT> = frame![col![1, 2, 3, NA, 4], col![1., 2., 3., NA, 4.],];
        assert_eq!(f.get(IntT), &col![1, 2, 3, NA, 4]);
    }

    #[test]
    fn test_threadsafe_frame() {
        let f = quickframe();
        let f2 = f.clone();
        let f3 = f.clone();
        ::std::thread::spawn(move || {
            let c = f2.get(IntT);
            c.build_index();
        });
        f.inner_join(&f3, IntT, IntT);
    }

    #[test]
    fn test_get_row() {
        use frunk;
        let f: Frame2<IntT, FloatT> = frame![col![1, 2], col![2., 3.]];
        {
            let row = f.get_row_hlist(0).unwrap();
            assert_eq!(
                row,
                frunk::HCons {
                    head: Some(&1),
                    tail: frunk::HCons {
                        head: Some(&2.),
                        tail: frunk::HNil
                    }
                }
            );
        }
        {
            let row = f.get_row(0).unwrap();
            assert_eq!(row, (Some(&1), Some(&2.)));
        }
    }

    #[test]
    fn test_replace_row() {
        let mut f = quickframe();
        // TODO This is very clunky. Making NamedColumns is too confusing and magic
        let newcol: NamedColumn<FloatT> = NamedColumn::new(col![5., NA, 3., 2., 0.]);
        f.replace(newcol.clone());
        assert_eq!(f.get(FloatT), &newcol);
    }

    #[test]
    #[should_panic]
    fn test_replace_row_panic() {
        let mut f = quickframe();
        let newcol: NamedColumn<FloatT> = NamedColumn::new(col![5., NA, 3., 2., 1., 0.]);
        f.replace(newcol);
    }

    // TODO
    // #[test]
    // fn test_display() {
    //     let f: Frame1<IntT> = frame![col![1, 2, 3, NA, 4]];
    //     format!("{}", f);
    // }

    // TODO
    // #[test]
    // fn test_ad_hoc_iter() {
    //     let f = quickframe();
    //     let iter = f.iter_rows::<(IntT, FloatT, IntT)>();
    //     assert_eq!(iter.next(), Some((Some(&1), Some(&5.), Some("this,'\"".to_string()))));
    // }
}
