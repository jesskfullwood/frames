use std::marker::PhantomData;

use column::{Column, Mask};
use frame::ColId;

// ### HList struct defs ###

#[derive(PartialEq, Debug, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub struct HNil;

#[derive(PartialEq, Debug, Clone)]
pub struct HCons<H: ColId, T> {
    pub head: Column<H::Output>,
    pub tail: T,
}

pub struct Here {
    _priv: (),
}

pub struct There<T> {
    _marker: PhantomData<T>,
}

#[derive(Debug, Deserialize, Copy, Clone)]
pub struct Product<T, U>(pub(crate) T, pub(crate) U);

// ### HList ###

pub trait HList: Sized {
    type Product;
    const SIZE: usize;
    const IS_ROOT: bool;

    #[inline]
    fn size(&self) -> usize {
        Self::SIZE
    }

    fn addcol<T: ColId>(self, head: impl Into<Column<T::Output>>) -> HCons<T, Self> {
        HCons {
            head: head.into(),
            tail: self,
        }
    }
}

impl HList for HNil {
    type Product = ();
    const SIZE: usize = 0;
    const IS_ROOT: bool = true;
}

impl<Head, Tail> HList for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList,
{
    type Product = Product<Option<Head::Output>, Tail::Product>;
    const SIZE: usize = 1 + <Tail as HList>::SIZE;
    const IS_ROOT: bool = false;
}

// ### HListExt ###

pub trait HListExt: HList {
    fn get_names(&self) -> Vec<&'static str>;
}

impl<Head, Tail> HListExt for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HListExt,
{
    fn get_names(&self) -> Vec<&'static str> {
        let mut ret = self.tail.get_names();
        ret.push(Head::NAME);
        ret
    }
}

impl HListExt for HNil {
    fn get_names(&self) -> Vec<&'static str> {
        Vec::new()
    }
}

// ### HListClonable ###

pub trait HListClonable: HList {
    fn copy_locs(&self, locs: &[usize]) -> Self;
    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self;
    fn filter_mask(&self, mask: &Mask) -> Self;
}

impl<Head, Tail> HListClonable for HCons<Head, Tail>
where
    Head: ColId,
    Head::Output: Clone,
    Tail: HListClonable,
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
    fn get(&self) -> &Column<S::Output>;
}

impl<Col: ColId, Tail> Selector<Col, Here> for HCons<Col, Tail> {
    fn get(&self) -> &Column<Col::Output> {
        &self.head
    }
}

impl<Head, Tail, FromTail, TailIndex> Selector<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
    Tail: Selector<FromTail, TailIndex>,
{
    fn get(&self) -> &Column<FromTail::Output> {
        self.tail.get()
    }
}

// ### Extractor ###

pub trait Extractor<Target: ColId, Index> {
    type Remainder: HList;
    fn extract(self) -> (Column<Target::Output>, Self::Remainder);
}

impl<Head: ColId, Tail: HList> Extractor<Head, Here> for HCons<Head, Tail> {
    type Remainder = Tail;

    fn extract(self) -> (Column<Head::Output>, Self::Remainder) {
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

    fn extract(self) -> (Column<FromTail::Output>, Self::Remainder) {
        let (target, tail_remainder): (
            Column<FromTail::Output>,
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
    fn replace(&mut self, newcol: Column<Target::Output>);
}

impl<Head, Tail> Replacer<Head, Here> for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList,
{
    fn replace(&mut self, newcol: Column<Head::Output>) {
        self.head = newcol;
    }
}

impl<Head, Tail, FromTail, TailIndex> Replacer<FromTail, There<TailIndex>> for HCons<Head, Tail>
where
    Head: ColId,
    FromTail: ColId,
    Tail: HList + Replacer<FromTail, TailIndex>,
{
    fn replace(&mut self, newcol: Column<FromTail::Output>) {
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

// ### Insertable

pub trait Insertable: HList {
    fn empty() -> Self;
    unsafe fn insert(&mut self, product: Self::Product);
}

impl<Head, Tail> Insertable for HCons<Head, Tail>
where
    Head: ColId,
    Tail: HList + Insertable,
{
    fn empty() -> Self {
        <Tail as Insertable>::empty().addcol(Vec::new())
    }
    unsafe fn insert(&mut self, product: Self::Product) {
        let Product(val, rest) = product;
        self.head.push(val);
        self.tail.insert(rest);
    }
}

impl Insertable for HNil {
    fn empty() -> Self {
        HNil
    }
    unsafe fn insert(&mut self, _product: ()) {}
}

// ### Transformer ###

pub trait Transformer {
    type Flattened;
    fn nest(flat: Self::Flattened) -> Self;
    fn flatten(self) -> Self::Flattened;
}

// TODO implement for other ProductN (with macro)
type Product3<T1, T2, T3> = Product<T3, Product<T2, Product<T1, ()>>>;
type Product2<T1, T2> = Product<T2, Product<T1, ()>>;
type Product1<T1> = Product<T1, ()>;

impl<T1, T2, T3> Transformer for Product3<T1, T2, T3> {
    type Flattened = (T1, T2, T3);
    fn nest(flat: Self::Flattened) -> Self {
        let (t1, t2, t3) = flat;
        Product(t3, Product(t2, Product(t1, ())))
    }
    fn flatten(self) -> Self::Flattened {
        let Product(t3, Product(t2, Product(t1, ()))) = self;
        (t1, t2, t3)
    }
}

impl<T1, T2> Transformer for Product2<T1, T2> {
    type Flattened = (T1, T2);
    fn nest(flat: Self::Flattened) -> Self {
        let (t1, t2) = flat;
        Product(t2, Product(t1, ()))
    }
    fn flatten(self) -> Self::Flattened {
        let Product(t2, Product(t1, ())) = self;
        (t1, t2)
    }
}

impl<T1> Transformer for Product1<T1> {
    type Flattened = (T1,);
    fn nest(flat: Self::Flattened) -> Self {
        let (t1,) = flat;
        Product(t1, ())
    }
    fn flatten(self) -> Self::Flattened {
        let Product(t1, ()) = self;
        (t1,)
    }
}
