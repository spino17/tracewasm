use crate::error::TracewasmUtilsError;
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash, marker::PhantomData};

#[derive(Debug)]
pub struct InternedId<T>(u16, PhantomData<T>);

impl<T> InternedId<T> {
    pub fn raw(&self) -> u16 {
        self.0
    }
}

impl<T> Clone for InternedId<T> {
    fn clone(&self) -> Self {
        InternedId(self.0, PhantomData)
    }
}

impl<T> Copy for InternedId<T> {}

pub trait Capacity {
    fn val() -> u64;
}

impl Capacity for u16 {
    fn val() -> u64 {
        u16::MAX as u64
    }
}

impl Capacity for u32 {
    fn val() -> u64 {
        u32::MAX as u64
    }
}

impl Capacity for u64 {
    fn val() -> u64 {
        u64::MAX
    }
}

pub struct Interner<T, C> {
    values: Vec<T>,
    reverse_map: FxHashMap<T, InternedId<T>>,
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

    pub fn intern(&mut self, val: T) -> Result<InternedId<T>, TracewasmUtilsError> {
        if let Some(id) = self.reverse_map.get(&val) {
            Ok(*id)
        } else {
            if self.values.len() >= C::val() as usize {
                return Err(TracewasmUtilsError::ToManyUniqueValues {
                    needed: self.values.len() as u32 + 1,
                    limit: C::val(),
                });
            }

            let id = InternedId(self.values.len() as u16, PhantomData);

            self.values.push(val.clone());
            self.reverse_map.insert(val, id);

            Ok(id)
        }
    }

    pub fn value(&self, id: InternedId<T>) -> &T {
        &self.values[id.0 as usize]
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
