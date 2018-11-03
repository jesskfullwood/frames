use column::{ColId, Mask, Column};

use std::fmt::{Debug, Display};

pub use frunk::hlist::HCons;
use frunk::hlist::{HList, HNil};
pub use frunk::indices::{Here, There};

// This module defines traits and methods on top of the `frunk` HList. For a good intro to HLists, see
// https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/

pub type ColCons<C, Tail> = HCons<Column<C>, Tail>;

// ### HListExt ###

pub trait HListExt: HList {
    type Product;
    fn get_names(&self, names: &mut Vec<&'static str>);
}

impl<Col, Tail> HListExt for ColCons<Col, Tail>
where
    Col: ColId,
    Tail: HListExt,
{
    type Product = HCons<Option<Col::Output>, Tail::Product>;
    fn get_names(&self, names: &mut Vec<&'static str>) {
        names.push(Col::NAME);
        self.tail.get_names(names);
    }
}

impl HListExt for HNil {
    type Product = HNil;
    fn get_names(&self, _names: &mut Vec<&'static str>) {}
}

// ### HListClonable ###

pub trait HListClonable: HList {
    fn copy_locs(&self, locs: &[usize]) -> Self;
    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self;
    fn filter_mask(&self, mask: &Mask) -> Self;
}

impl<Col, Tail> HListClonable for ColCons<Col, Tail>
where
    Col: ColId,
    Col::Output: Clone,
    Tail: HListClonable,
{
    fn copy_locs(&self, locs: &[usize]) -> Self {
        ColCons {
            head: Column::new(self.head.copy_locs(locs)),
            tail: self.tail.copy_locs(locs),
        }
    }

    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self {
        ColCons {
            head: Column::new(self.head.copy_locs_opt(locs)),
            tail: self.tail.copy_locs_opt(locs),
        }
    }

    fn filter_mask(&self, mask: &Mask) -> Self {
        ColCons {
            head: Column::new(self.head.filter_mask(mask)),
            tail: self.tail.filter_mask(mask),
        }
    }
}

impl HListClonable for HNil {
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

// ### Appender ###

pub trait Appender<T> {
    type FromRoot;
    fn append(self, c: T) -> Self::FromRoot;
}

impl<T, Head, Tail> Appender<T> for HCons<Head, Tail>
where
    Tail: Appender<T>,
{
    type FromRoot = HCons<Head, Tail::FromRoot>;
    fn append(self, c: T) -> Self::FromRoot {
        HCons {
            head: self.head,
            tail: self.tail.append(c),
        }
    }
}

impl<T> Appender<T> for HNil {
    type FromRoot = HCons<T, HNil>;
    fn append(self, c: T) -> Self::FromRoot {
        HCons {
            head: c,
            tail: self,
        }
    }
}

// ### Concat ###

pub trait Concat<C> {
    type Combined: HList;
    fn concat(self, other: C) -> Self::Combined;
}

impl<C: HList> Concat<C> for HNil {
    type Combined = C;
    fn concat(self, other: C) -> Self::Combined {
        other
    }
}

impl<Head, Tail, C> Concat<C> for ColCons<Head, Tail>
where
    Head: ColId,
    Tail: Concat<C>,
{
    type Combined = ColCons<Head, <Tail as Concat<C>>::Combined>;
    fn concat(self, other: C) -> Self::Combined {
        ColCons {
            head: self.head,
            tail: self.tail.concat(other),
        }
    }
}

// ### Replacer ###

pub trait Replacer<Target: ColId, Index> {
    fn replace(&mut self, newcol: Column<Target>);
}

impl<Col, Tail> Replacer<Col, Here> for ColCons<Col, Tail>
where
    Col: ColId,
    Tail: HList,
{
    fn replace(&mut self, newcol: Column<Col>) {
        self.head = newcol;
    }
}

impl<Col, Tail, FromTail, TailIndex> Replacer<FromTail, There<TailIndex>> for ColCons<Col, Tail>
where
    Col: ColId,
    FromTail: ColId,
    Tail: HList + Replacer<FromTail, TailIndex>,
{
    fn replace(&mut self, newcol: Column<FromTail>) {
        self.tail.replace(newcol)
    }
}

// ### Mapper ###

pub trait Mapper<Target: ColId, NewCol: ColId, Index> {
    type Mapped: HList;
    fn map_replace_all<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&Target::Output>) -> Option<<NewCol as ColId>::Output>;

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Target::Output) -> <NewCol as ColId>::Output;
}

impl<Head, Tail, NewCol> Mapper<Head, NewCol, Here> for ColCons<Head, Tail>
where
    Head: ColId,
    NewCol: ColId,
    Tail: HList,
{
    type Mapped = ColCons<NewCol, Tail>;
    fn map_replace_all<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&Head::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        ColCons {
            head: Column::new(self.head.map_null(func)),
            tail: self.tail,
        }
    }

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Head::Output) -> <NewCol as ColId>::Output,
    {
        ColCons {
            head: Column::new(self.head.map(func)),
            tail: self.tail,
        }
    }
}

impl<Col, Tail, NewCol, FromTail, TailIndex> Mapper<FromTail, NewCol, There<TailIndex>>
    for ColCons<Col, Tail>
where
    Col: ColId,
    FromTail: ColId,
    NewCol: ColId,
    Tail: HList + Mapper<FromTail, NewCol, TailIndex>,
{
    type Mapped = ColCons<Col, <Tail as Mapper<FromTail, NewCol, TailIndex>>::Mapped>;

    fn map_replace_all<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&FromTail::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        ColCons {
            head: self.head,
            tail: self.tail.map_replace_all(func),
        }
    }

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&FromTail::Output) -> <NewCol as ColId>::Output,
    {
        ColCons {
            head: self.head,
            tail: self.tail.map_replace(func),
        }
    }
}

// ### Stringify ###

pub trait Stringify {
    fn stringify(&self) -> Vec<String>;
}

impl<Head, Tail> Stringify for HCons<Head, Tail>
where
    Head: Debug,
    Tail: Stringify,
{
    fn stringify(&self) -> Vec<String> {
        let mut out = self.tail.stringify();
        out.push(format!("{:?}", self.head));
        out
    }
}

impl Stringify for HNil {
    fn stringify(&self) -> Vec<String> {
        Vec::new()
    }
}

// ### RowHList ###

pub trait RowHList<'a> {
    type ProductOptRef;
    fn get_row(&'a self, index: usize) -> Self::ProductOptRef;
}

impl<'a, Head, Tail> RowHList<'a> for ColCons<Head, Tail>
where
    Head: ColId,
    Head::Output: 'a,
    Tail: RowHList<'a>,
{
    // TODO this could be tidied up a lot with GATs:
    // https://github.com/rust-lang/rust/issues/44265
    type ProductOptRef = HCons<Option<&'a Head::Output>, Tail::ProductOptRef>;
    fn get_row(&'a self, index: usize) -> Self::ProductOptRef {
        // NOTE: assumes the provided index is valid. This should be checked by the parent frame
        HCons {
            head: self.head.get(index).unwrap(),
            tail: self.tail.get_row(index),
        }
    }
}

impl<'a> RowHList<'a> for HNil {
    type ProductOptRef = HNil;
    fn get_row(&self, _index: usize) -> Self::ProductOptRef {
        HNil
    }
}

// ### Transformer ###

pub trait Transformer {
    type Flattened;
    fn nest(flat: Self::Flattened) -> Self;
    fn flatten(self) -> Self::Flattened;
}

// ### Insertable

pub trait Insertable {
    type ProductOpt;
    fn empty() -> Self;
    fn insert(&mut self, product_opt: Self::ProductOpt);
}

impl<Col, Tail> Insertable for ColCons<Col, Tail>
where
    Col: ColId,
    Tail: HList + Insertable,
{
    type ProductOpt = HCons<Option<Col::Output>, Tail::ProductOpt>;
    fn empty() -> Self {
        <Tail as Insertable>::empty().prepend(Column::empty())
    }
    fn insert(&mut self, product: Self::ProductOpt) {
        let HCons {
            head: optval,
            tail: rest,
        } = product;
        self.head.insert(optval);
        self.tail.insert(rest);
    }
}

impl Insertable for HNil {
    type ProductOpt = HNil;
    fn empty() -> Self {
        HNil
    }
    fn insert(&mut self, _product: Self::ProductOpt) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stringify() {
        let cons = HCons {
            head: 10,
            tail: HCons {
                head: "hello",
                tail: HNil,
            },
        };
        let words = cons.stringify();
        assert_eq!(words, vec!["hello".to_string(), "10".to_string()]);
    }
}
