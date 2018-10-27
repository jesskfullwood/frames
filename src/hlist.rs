use column::{ColId, Mask, NamedColumn};

use std::fmt::{Debug, Display};

pub use frunk::hlist::HCons as HConsFrunk;
use frunk::hlist::{HList, HNil};
pub use frunk::indices::{Here, There};

// This module defines traits and methods on top of the `frunk` HList. For a good intro to HLists, see
// https://beachape.com/blog/2017/03/12/gentle-intro-to-type-level-recursion-in-Rust-from-zero-to-frunk-hlist-sculpting/

pub type HCons<C, Tail> = HConsFrunk<NamedColumn<C>, Tail>;

// ### HListExt ###

pub trait HListExt: HList {
    type Product;
    fn get_names(&self, names: &mut Vec<&'static str>);
}

impl<Col, Tail> HListExt for HCons<Col, Tail>
where
    Col: ColId,
    Tail: HListExt,
{
    type Product = HConsFrunk<Option<Col::Output>, Tail::Product>;
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

impl<Col, Tail> HListClonable for HCons<Col, Tail>
where
    Col: ColId,
    Col::Output: Clone,
    Tail: HListClonable,
{
    fn copy_locs(&self, locs: &[usize]) -> Self {
        HCons {
            head: NamedColumn::new(self.head.copy_locs(locs)),
            tail: self.tail.copy_locs(locs),
        }
    }

    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self {
        HCons {
            head: NamedColumn::new(self.head.copy_locs_opt(locs)),
            tail: self.tail.copy_locs_opt(locs),
        }
    }

    fn filter_mask(&self, mask: &Mask) -> Self {
        HCons {
            head: NamedColumn::new(self.head.filter_mask(mask)),
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

impl<T, Head, Tail> Appender<T> for HConsFrunk<Head, Tail>
where
    Tail: Appender<T>,
{
    type FromRoot = HConsFrunk<Head, Tail::FromRoot>;
    fn append(self, c: T) -> Self::FromRoot {
        HConsFrunk {
            head: self.head,
            tail: self.tail.append(c),
        }
    }
}

impl<T> Appender<T> for HNil {
    type FromRoot = HConsFrunk<T, HNil>;
    fn append(self, c: T) -> Self::FromRoot {
        HConsFrunk {
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

impl<Head, Tail, C> Concat<C> for HCons<Head, Tail>
where
    Head: ColId,
    Tail: Concat<C>,
{
    type Combined = HCons<Head, <Tail as Concat<C>>::Combined>;
    fn concat(self, other: C) -> Self::Combined {
        HCons {
            head: self.head,
            tail: self.tail.concat(other),
        }
    }
}

// ### Replacer ###

pub trait Replacer<Target: ColId, Index> {
    fn replace(&mut self, newcol: NamedColumn<Target>);
}

impl<Col, Tail> Replacer<Col, Here> for HCons<Col, Tail>
where
    Col: ColId,
    Tail: HList,
{
    fn replace(&mut self, newcol: NamedColumn<Col>) {
        self.head = newcol;
    }
}

impl<Col, Tail, FromTail, TailIndex> Replacer<FromTail, There<TailIndex>> for HCons<Col, Tail>
where
    Col: ColId,
    FromTail: ColId,
    Tail: HList + Replacer<FromTail, TailIndex>,
{
    fn replace(&mut self, newcol: NamedColumn<FromTail>) {
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

impl<Head, Tail, NewCol> Mapper<Head, NewCol, Here> for HCons<Head, Tail>
where
    Head: ColId,
    NewCol: ColId,
    Tail: HList,
{
    type Mapped = HCons<NewCol, Tail>;
    fn map_replace_all<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&Head::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        HCons {
            head: NamedColumn::new(self.head.map_null(func)),
            tail: self.tail,
        }
    }

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Head::Output) -> <NewCol as ColId>::Output,
    {
        HCons {
            head: NamedColumn::new(self.head.map(func)),
            tail: self.tail,
        }
    }
}

impl<Col, Tail, NewCol, FromTail, TailIndex> Mapper<FromTail, NewCol, There<TailIndex>>
    for HCons<Col, Tail>
where
    Col: ColId,
    FromTail: ColId,
    NewCol: ColId,
    Tail: HList + Mapper<FromTail, NewCol, TailIndex>,
{
    type Mapped = HCons<Col, <Tail as Mapper<FromTail, NewCol, TailIndex>>::Mapped>;

    fn map_replace_all<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(Option<&FromTail::Output>) -> Option<<NewCol as ColId>::Output>,
    {
        HCons {
            head: self.head,
            tail: self.tail.map_replace_all(func),
        }
    }

    fn map_replace<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&FromTail::Output) -> <NewCol as ColId>::Output,
    {
        HCons {
            head: self.head,
            tail: self.tail.map_replace(func),
        }
    }
}

// ### Stringify ###

pub trait Stringify {
    fn stringify(&self) -> Vec<String>;
}

impl<Head, Tail> Stringify for HConsFrunk<Head, Tail>
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

impl<'a, Head, Tail> RowHList<'a> for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: 'a,
    Tail: RowHList<'a>,
{
    // TODO this could be tidied up a lot with GATs:
    // https://github.com/rust-lang/rust/issues/44265
    type ProductOptRef = HConsFrunk<Option<&'a Head::Output>, Tail::ProductOptRef>;
    fn get_row(&'a self, index: usize) -> Self::ProductOptRef {
        // NOTE: assumes the provided index is valid. This should be checked by the parent frame
        HConsFrunk {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stringify() {
        let cons = HConsFrunk {
            head: 10,
            tail: HConsFrunk {
                head: "hello",
                tail: HNil,
            },
        };
        let words = cons.stringify();
        assert_eq!(words, vec!["hello".to_string(), "10".to_string()]);
    }
}
