use frame::{Frame, HCons, HNil};

macro_rules! frame_alias {
    ($($frames:ident),+ -> $typFirst:ident, $($typsNext:ident),*) => { // start things off
        pub type Frame0 = Frame<HNil>;
        frame_alias!($($frames)* -> [ $typFirst ] $($typsNext)*);
    };
    ($frame:ident $($frames:ident)* -> [$($typsCur:ident)+] $typNext:ident $($typsFut:ident)*) => {
        pub type $frame<$($typsCur,)*> = Frame<cons!([$($typsCur)*])>;
        frame_alias!($($frames)* -> [ $($typsCur)* $typNext ] $($typsFut)*);
    };
    ($frame:ident -> [$($typsCur:ident)+]) => {
        pub type $frame<$($typsCur,)*> = Frame<cons!([$($typsCur)*])>;
    }
}

macro_rules! cons_disp {
    () => {};
    ($typ: ident) => {
        HCons<$typ, HNil>
    };
    ($typFront:ident $($typs:ident)+) => {
        HCons<$typFront, cons_disp!($($typs)*)>
    }
}

macro_rules! cons {
    ([] $($typs:ident)*) => {
        cons_disp!($($typs)*)
    };
    ([$first:ident $($rest:ident)*] $($reversed:ident)*) => {
        cons!([$($rest)*] $first $($reversed)*)
    };
}

frame_alias!(
    Frame1,
    Frame2,
    Frame3,
    Frame4,
    Frame5,
    Frame6,
    Frame7,
    Frame8,
    Frame9,
    Frame10,
    Frame11,
    Frame12,
    Frame13,
    Frame14,
    Frame15,
    Frame16,
    Frame17,
    Frame18,
    Frame19,
    Frame20,
    Frame21,
    Frame22,
    Frame23,
    Frame24,
    Frame25,
    Frame26,
    Frame27,
    Frame28,
    Frame29,
    Frame30,
    Frame31,
    Frame32
    ->
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T19,
    T20,
    T21,
    T22,
    T23,
    T24,
    T25,
    T26,
    T27,
    T28,
    T29,
    T30,
    T31,
    T32
);

