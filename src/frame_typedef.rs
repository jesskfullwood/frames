use frame::Frame;
use frunk::HNil;
use hlist::{HCons, HConsFrunk, Transformer};

pub type Frame0 = Frame<HNil>;
pub type Frame1<T1> = Frame<HCons<T1, HNil>>;
pub type Frame2<T1, T2> = Frame<HCons<T1, HCons<T2, HNil>>>;
pub type Frame3<T1, T2, T3> = Frame<HCons<T1, HCons<T2, HCons<T3, HNil>>>>;

macro_rules! frame_alias {
    ($($frames:ident,)+ -> $typFirst:ident, $($typsNext:ident,)*) => {
        // start things off
        frame_alias!($($frames)* -> [ $typFirst ] $($typsNext)*);
    };
    ($frame:ident $($frames:ident)* -> [$($typsCur:ident)+] $typNext:ident $($typsFut:ident)*) => {
        frame_def!($frame $($typsCur)*);
        frame_alias!($($frames)* -> [ $($typsCur)* $typNext ] $($typsFut)*);
        transformer_impl!($($typsCur)+);
    };
    ($frame:ident -> [$($typsCur:ident)+]) => {
        frame_def!($frame $($typsCur)+);
        transformer_impl!($($typsCur)+);
    }
}

macro_rules! transformer_impl {
    ($($typs:ident)+) => {
        impl<$($typs,)+> Transformer for macro_revargs!(cons_frunk $($typs)+) {
            type Flattened = ($($typs,)+);
            #[allow(non_snake_case)]
            fn nest(flat: Self::Flattened) -> Self {
                let ($($typs,)+) = flat;
                macro_revargs!(hlist_structure $($typs)+)
            }
            #[allow(non_snake_case)]
            fn flatten(self) -> Self::Flattened {
                let macro_revargs!(hlist_structure $($typs)+) = self;
                ($($typs,)+)
            }
        }
    }
}

macro_rules! macro_revargs {
    ($macro:ident $($args:ident)*) => {
        macro_revargs!($macro [$($args)*])
    };
    ($macro:ident [$reverse:ident $($to_reverse:ident)*] $($reversed:ident)*) => {
        macro_revargs!($macro [$($to_reverse)*] $reverse $($reversed)*)
    };
    ($macro:ident [] $($reversed:ident)*) => {
        $macro!($($reversed)*)
    };
}

macro_rules! frame_def {
    ($frame:ident $($typsCur:ident)+) => {
        pub type $frame<$($typsCur,)+> = Frame<macro_revargs!(cons $($typsCur)+)>;
    }
}

macro_rules! cons {
    ($typ: ident) => {
        HCons<$typ, HNil>
    };
    ($typ_front:ident $($typs:ident)+) => {
        HCons<$typ_front, cons!($($typs)+)>
    }
}

macro_rules! cons_frunk {
    ($typ: ident) => {
        HConsFrunk<$typ, HNil>
    };
    ($typ_front:ident $($typs:ident)+) => {
        HConsFrunk<$typ_front, cons_frunk!($($typs)+)>
    }
}

macro_rules! hlist_structure {
    ($typ: ident) => {
        HConsFrunk { head: $typ, tail: HNil }
    };
    ($typ_front:ident $($typs:ident)+) => {
        HConsFrunk { head: $typ_front, tail: hlist_structure!($($typs)+) }
    }
}

// frame_alias!(
//     Frame1,
//     Frame2,
//     Frame3,
//     Frame4,
//     Frame5,
//     Frame6,
//     Frame7,
//     Frame8,
//     Frame9,
//     Frame10,
//     Frame11,
//     Frame12,
//     Frame13,
//     Frame14,
//     Frame15,
//     Frame16,
//     Frame17,
//     Frame18,
//     Frame19,
//     Frame20,
//     Frame21,
//     Frame22,
//     Frame23,
//     Frame24,
//     Frame25,
//     Frame26,
//     Frame27,
//     Frame28,
//     Frame29,
//     Frame30,
//     Frame31,
//     Frame32,
//     ->
//     T1,
//     T2,
//     T3,
//     T4,
//     T5,
//     T6,
//     T7,
//     T8,
//     T9,
//     T10,
//     T11,
//     T12,
//     T13,
//     T14,
//     T15,
//     T16,
//     T17,
//     T18,
//     T19,
//     T20,
//     T21,
//     T22,
//     T23,
//     T24,
//     T25,
//     T26,
//     T27,
//     T28,
//     T29,
//     T30,
//     T31,
//     T32,
// );
