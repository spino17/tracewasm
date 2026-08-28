use crate::value::ConstValue;
use std::ops::{Deref, DerefMut};
use tracewasm_utils::interner::{InternedId, Interner};

#[derive(Default)]
pub struct StrInterner(Interner<String, u32>);

impl Deref for StrInterner {
    type Target = Interner<String, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for StrInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct StrId(pub InternedId<String, u32>);

impl From<InternedId<String, u32>> for StrId {
    fn from(value: InternedId<String, u32>) -> Self {
        StrId(value)
    }
}

#[derive(Default)]
pub struct ConstInterner(Interner<ConstValue, u32>);

impl Deref for ConstInterner {
    type Target = Interner<ConstValue, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ConstInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ConstId(InternedId<ConstValue, u32>);

impl From<InternedId<ConstValue, u32>> for ConstId {
    fn from(value: InternedId<ConstValue, u32>) -> Self {
        ConstId(value)
    }
}
