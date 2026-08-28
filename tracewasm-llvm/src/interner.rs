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

pub struct StrId(InternedId<String>);

impl From<InternedId<String>> for StrId {
    fn from(value: InternedId<String>) -> Self {
        StrId(value)
    }
}

pub enum Const {}

#[derive(Default)]
pub struct ConstInterner(Interner<Const, u32>);

impl Deref for ConstInterner {
    type Target = Interner<Const, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ConstInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct ConstId(InternedId<Const>);

impl From<InternedId<Const>> for ConstId {
    fn from(value: InternedId<Const>) -> Self {
        ConstId(value)
    }
}
