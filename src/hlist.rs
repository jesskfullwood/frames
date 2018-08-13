use column::{ColId, Column, Mask};
pub use frunk::hlist::{HCons as HConsFrunk, HList, HNil};
pub use frunk::indices::{Here, There};

pub type HCons<C, Tail> = HConsFrunk<Column<C>, Tail>;

// ### HListExt ###

pub trait HListExt: HList {
    type Product;
    fn get_names(&self) -> Vec<&'static str>;
}

impl<Col, Tail> HListExt for HCons<Col, Tail>
where
    Col: ColId,
    Tail: HListExt,
{
    type Product = HConsFrunk<Option<Col::Output>, Tail::Product>;
    fn get_names(&self) -> Vec<&'static str> {
        let mut ret = self.tail.get_names();
        ret.push(Col::NAME);
        ret
    }
}

impl HListExt for HNil {
    type Product = HNil;
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

impl<Col, Tail> HListClonable for HCons<Col, Tail>
where
    Col: ColId,
    Col::Output: Clone,
    Tail: HListClonable,
{
    fn copy_locs(&self, locs: &[usize]) -> Self {
        HCons {
            head: Column::new(self.head.copy_locs(locs)),
            tail: self.tail.copy_locs(locs),
        }
    }

    fn copy_locs_opt(&self, locs: &[Option<usize>]) -> Self {
        HCons {
            head: Column::new(self.head.copy_locs_opt(locs)),
            tail: self.tail.copy_locs_opt(locs),
        }
    }

    fn filter_mask(&self, mask: &Mask) -> Self {
        HCons {
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

// ### Replacer ###

pub trait Replacer<Target: ColId, Index> {
    fn replace(&mut self, newcol: Column<Target>);
}

impl<Col, Tail> Replacer<Col, Here> for HCons<Col, Tail>
where
    Col: ColId,
    Tail: HList,
{
    fn replace(&mut self, newcol: Column<Col>) {
        self.head = newcol;
    }
}

impl<Col, Tail, FromTail, TailIndex> Replacer<FromTail, There<TailIndex>> for HCons<Col, Tail>
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
            head: Column::new(self.head.map(func)),
            tail: self.tail,
        }
    }

    fn map_replace_notnull<F>(self, func: F) -> Self::Mapped
    where
        F: Fn(&Head::Output) -> <NewCol as ColId>::Output,
    {
        HCons {
            head: Column::new(self.head.map_notnull(func)),
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

pub trait Insertable: HListExt {
    fn empty() -> Self;
    unsafe fn insert(&mut self, product: Self::Product);
}

impl<Col, Tail> Insertable for HCons<Col, Tail>
where
    Col: ColId,
    Tail: Insertable,
{
    fn empty() -> Self {
        <Tail as Insertable>::empty().prepend(Vec::new().into())
    }
    unsafe fn insert(&mut self, product: Self::Product) {
        let HConsFrunk {
            head: val,
            tail: rest,
        } = product;
        self.head.push(val);
        self.tail.insert(rest);
    }
}

impl Insertable for HNil {
    fn empty() -> Self {
        HNil
    }
    unsafe fn insert(&mut self, _product: HNil) {}
}

// ### Transformer ###

pub trait Transformer {
    type Flattened;
    fn nest(flat: Self::Flattened) -> Self;
    fn flatten(self) -> Self::Flattened;
}
