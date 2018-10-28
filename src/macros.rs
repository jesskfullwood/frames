
// TODO this only works for single idents, ie "my string column" is not allowed
#[macro_export]
macro_rules! define_col {
    ($tyname:ident, $typ:ty) => {
        define_col!($tyname, $typ, $tyname);
    };
    ($tyname:ident, $typ:ty, $name:ident) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $tyname;
        impl $crate::column::ColId for $tyname {
            const NAME: &'static str = stringify!($name);
            type Output = $typ;
        }
    };
}

#[macro_export]
macro_rules! define_frame {
    // ($tyname:ident, $typ:ty) => {
    //     define_frame!($tyname, $typ, $tyname);
    // };
    ($tyname:ident, $typ:ty, $name:ident) => {
        type MyFrame = Frame3<Name, Age, Favourite
    };
}

#[macro_export]
macro_rules! frame {
    () => {
        $crate::Frame::empty()
    };
    ($([$($x:tt),* $(,)*]),+ $(,)*) => {
        {
            let mut f = $crate::Frame::empty();
            $(
                let row = ($(wrap_val!($x),)*);
                f.insert_row(row);
            )*;
            f
        }
    }
}

#[allow(unused_macros)]
#[macro_use]
macro_rules! wrap_val {
    (NA) => {
        None
    };
    ($val:tt) => {
        Some($val)
    };
}

#[macro_export]
macro_rules! frame_col {
    () => {
        $crate::Frame::empty()
    };
    ($([$($x:tt),* $(,)*]),+ $(,)*) => {
        {
            let f = $crate::Frame::new();
            $(
                let c: $crate::Column<_> = col![$($x,)*];
                let f = f.add(c).unwrap();
            )*;
            f
        }
    }
}
