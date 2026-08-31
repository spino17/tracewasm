use crate::value::{ConstValue, Type, TypeDisplay};
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

impl ConstId {
    pub(crate) fn raw(&self) -> InternedId<ConstValue, u32> {
        self.0
    }
}

impl From<InternedId<ConstValue, u32>> for ConstId {
    fn from(value: InternedId<ConstValue, u32>) -> Self {
        ConstId(value)
    }
}

#[derive(Default)]
pub struct TyInterner(Interner<Type, u32>);

impl Deref for TyInterner {
    type Target = Interner<Type, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TyInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TyInterner {
    /// Resolves `id` and borrows it against this pool for rendering, which is the
    /// only way an id becomes the type it names.
    pub fn display(&self, id: TyId) -> TypeDisplay<'_> {
        self.value(id.raw()).display(self)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TyId(pub InternedId<Type, u32>);

impl TyId {
    pub(crate) fn raw(&self) -> InternedId<Type, u32> {
        self.0
    }
}

impl From<InternedId<Type, u32>> for TyId {
    fn from(value: InternedId<Type, u32>) -> Self {
        TyId(value)
    }
}
