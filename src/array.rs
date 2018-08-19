use std::ops::Index;

/// NewType wrapper around Vec
#[derive(Debug, Clone)]
pub(crate) struct Array<T>(Vec<T>);

impl<T> Array<T> {
    #[inline]
    pub(crate) fn new(data: Vec<T>) -> Self {
        Array(data)
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    #[inline]
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Index<usize> for Array<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
