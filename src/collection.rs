use *;

#[derive(Clone)]
pub struct Collection<T> {
    pub data: Array<T>,
    index: RefCell<Option<IndexMap<T>>>,
}

impl<T> From<Array<T>> for Collection<T> {
    fn from(arr: Array<T>) -> Collection<T> {
        Collection::new(arr)
    }
}

impl<T: PartialEq> PartialEq for Collection<T> {
    // We don't care if the indexes are the same
    fn eq(&self, other: &Self) -> Bool {
        self.data == other.data
    }
}

impl<T: Debug> Debug for Collection<T> {
    fn fmt(&self, f: &mut Formatter) -> StdResult<(), std::fmt::Error> {
        write!(
            f,
            "Collection {{ indexed: {}, vals: {:?} }}",
            self.index.borrow().is_some(),
            self.data
        )
    }
}

impl<T> Collection<T> {
    pub fn new(data: Array<T>) -> Collection<T> {
        Collection {
            data,
            index: RefCell::new(None),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn has_index(&self) -> Bool {
        self.index.borrow().is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.data.iter()
    }

    pub fn map<R>(&self, test: impl Fn(&T) -> R) -> Collection<R> {
        Collection::new(self.iter().map(test).collect())
    }
}

impl<A, T> PartialEq<A> for Collection<T>
where
    A: AsRef<[T]>,
    T: PartialEq,
{
    fn eq(&self, other: &A) -> bool {
        let slice: &[T] = other.as_ref();
        self.iter().zip(slice.iter()).all(|(l, r)| l == r)
    }
}

impl<T: Clone> Collection<T> {
    /// Create new Collection taking values from provided slice of indices
    pub(crate) fn copy_locs(&self, locs: &[usize]) -> Collection<T> {
        let data = locs.iter().map(|&ix| self.data[ix].clone()).collect();
        Collection::new(data)
    }

    pub fn apply_mask(&self, mask: &Mask) -> Self {
        assert_eq!(self.len(), mask.len());
        Collection::new(
            self.iter()
                .zip(mask.iter())
                .filter_map(|(v, &b)| if b { Some(v.clone()) } else { None })
                .collect(),
        )
    }
}

impl<T: Hash + Clone + Eq> Collection<T> {
    pub fn build_index(&self) {
        if self.has_index() {
            return;
        }
        let mut index = IndexMap::new();
        for (ix, d) in self.data.iter().enumerate() {
            let entry = index.entry(d.clone()).or_insert(Vec::new());
            entry.push(ix)
        }
        *self.index.borrow_mut() = Some(index);
    }

    pub(crate) fn inner_join_locs(&self, other: &Collection<T>) -> (Vec<usize>, Vec<usize>) {
        // TODO if "other" is already indexed, we can skip this step
        self.build_index();

        let borrow = self.index.borrow();
        let colix = borrow.as_ref().unwrap();
        let mut pair: Vec<(usize, usize)> = Vec::new();
        for (rix, val) in other.iter().enumerate() {
            if let Some(lixs) = colix.get(val) {
                lixs.iter().for_each(|&lix| pair.push((lix, rix)))
            }
        }
        pair.sort_unstable();
        let mut left = Vec::with_capacity(pair.len());
        let mut right = Vec::with_capacity(pair.len());
        pair.iter().for_each(|&(l, r)| {
            left.push(l);
            right.push(r);
        });
        (left, right)
    }
}

impl<T: Num + Copy> Collection<T> {
    // TODO big risk of overflow for ints
    // use some kind of bigint
    pub fn sum(&self) -> T {
        self.data.iter().fold(num::zero(), |acc, &v| acc + v)
    }
}

impl<T: Num + Copy + AsPrimitive<f64>> Collection<T> {
    pub fn mean(&self) -> f64 {
        let s: f64 = self.sum().as_();
        s / self.len() as f64
    }

    pub fn variance(&self) -> f64 {
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;
        self.data.iter().for_each(|n| {
            let n: f64 = n.as_();
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.len() as f64;
        sigmafxsqr / self.len() as f64 - mean * mean
    }

    pub fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn describe(&self) -> Describe {
        let mut min = std::f64::MAX;
        let mut max = std::f64::MIN;
        let mut sigmafxsqr: f64 = 0.;
        let mut sigmafx: f64 = 0.;

        self.data.iter().for_each(|n| {
            let n: f64 = n.as_();
            if n < min {
                min = n;
            }
            if n > max {
                max = n
            }
            sigmafxsqr += n * n;
            sigmafx += n;
        });
        let mean = sigmafx / self.len() as f64;
        let variance = sigmafxsqr / self.len() as f64 - mean * mean;
        Describe {
            len: self.len(),
            min,
            max,
            mean,
            stdev: variance.sqrt(),
        }
    }
}

impl Collection<Float> {
    pub(crate) fn as_ordered(&self) -> &Collection<OrdFloat> {
        unsafe { &*(self as *const Self as *const Collection<OrdFloat>) }
    }
}
