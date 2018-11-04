// TODO this only works for single idents, ie "my string column" is not allowed
#[macro_export]
macro_rules! define_col {
    ($($tyname:ident: $typ:ty),*) => {
        $(
            define_col!($tyname, $typ, $tyname);
        )*
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
                let row = ($(frame!(wrap $x),)*);
                f.push_row(row);
            )*;
            f
        }
    };
    (wrap NA) => {
        None
    };
    (wrap $val:tt) => {
        Some($val)
    }
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
                let c: $crate::Array<_> = col![$($x,)*];
                let f = f.add(c).unwrap();
            )*;
            f
        }
    }
}

// TODO document usage
#[macro_export]
macro_rules! col {
    ($($vals:tt),* $(,)*) => {
        {
            let mut v = Vec::new();
            $(
                let val = col!(wrap $vals);
                v.push(val);
            )*
                $crate::Array::from(v)
        }
    };
    (wrap NA) => {
        None
    };
    (wrap $val:tt) => {
        Some($val)
    }
}
