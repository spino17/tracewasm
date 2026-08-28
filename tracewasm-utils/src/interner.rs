use crate::error::TracewasmUtilsError;
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash, marker::PhantomData};

#[derive(Debug)]
pub struct InternedId<T, C>(C, PhantomData<T>);

impl<T, C: Capacity> InternedId<T, C> {
    pub fn raw(&self) -> C {
        self.0
    }
}

impl<T, C: Capacity> Clone for InternedId<T, C> {
    fn clone(&self) -> Self {
        InternedId(self.0, PhantomData)
    }
}

impl<T, C: Capacity> Copy for InternedId<T, C> {}

pub trait Capacity: Clone + Copy {
    fn val() -> u64;

    fn from_usize(val: usize) -> Self;

    fn to_usize(&self) -> usize;
}

impl Capacity for u16 {
    fn val() -> u64 {
        u16::MAX as u64
    }

    fn from_usize(val: usize) -> Self {
        val as u16
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Capacity for u32 {
    fn val() -> u64 {
        u32::MAX as u64
    }

    fn from_usize(val: usize) -> Self {
        val as u32
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Capacity for u64 {
    fn val() -> u64 {
        u64::MAX
    }

    fn from_usize(val: usize) -> Self {
        val as u64
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

pub struct Interner<T, C> {
    values: Vec<T>,
    reverse_map: FxHashMap<T, InternedId<T, C>>,
    phantom: PhantomData<C>,
}

impl<T, C> Default for Interner<T, C> {
    fn default() -> Self {
        Interner {
            values: vec![],
            reverse_map: FxHashMap::default(),
            phantom: PhantomData,
        }
    }
}

impl<T: Clone + PartialEq + Eq + Hash, C: Capacity> Interner<T, C> {
    pub fn new() -> Self {
        Interner {
            values: Vec::default(),
            reverse_map: FxHashMap::default(),
            phantom: PhantomData,
        }
    }

    pub fn intern(&mut self, val: T) -> Result<InternedId<T, C>, TracewasmUtilsError> {
        if let Some(id) = self.reverse_map.get(&val) {
            Ok(*id)
        } else {
            if self.values.len() >= C::val() as usize {
                return Err(TracewasmUtilsError::ToManyUniqueValues {
                    needed: self.values.len() as u32 + 1,
                    limit: C::val(),
                });
            }

            let id = InternedId(C::from_usize(self.values.len()), PhantomData);

            self.values.push(val.clone());
            self.reverse_map.insert(val, id);

            Ok(id)
        }
    }

    pub fn value(&self, id: InternedId<T, C>) -> &T {
        &self.values[id.0.to_usize()]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}
