use crate::error::TraceWasmError;
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash, marker::PhantomData};

#[derive(Debug)]
pub(crate) struct InternedId<T>(u16, PhantomData<T>);

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

pub(crate) trait TyToString {
    fn to_string() -> String;
}

#[derive(Default)]
pub(crate) struct Interner<T> {
    values: Vec<T>,
    reverse_map: FxHashMap<T, InternedId<T>>,
    max_size: u16,
}

impl<T: Clone + PartialEq + Eq + Hash + TyToString> Interner<T> {
    pub fn new(max_size: u16) -> Self {
        Interner {
            values: Vec::default(),
            reverse_map: FxHashMap::default(),
            max_size,
        }
    }

    pub fn intern(&mut self, val: T) -> Result<InternedId<T>, TraceWasmError> {
        if let Some(id) = self.reverse_map.get(&val) {
            Ok(*id)
        } else {
            if self.values.len() >= self.max_size as usize {
                return Err(TraceWasmError::ToManyUniqueValues {
                    what: T::to_string(),
                    needed: self.values.len() as u32 + 1,
                    limit: self.max_size as u32,
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
