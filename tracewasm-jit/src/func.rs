pub trait LLVMFuncParam {}

impl LLVMFuncParam for i32 {}
impl LLVMFuncParam for i64 {}
impl LLVMFuncParam for f32 {}
impl LLVMFuncParam for f64 {}

pub trait LLVMFuncParams {}

impl LLVMFuncParams for () {}
impl<A1: LLVMFuncParam> LLVMFuncParams for (A1,) {}
impl<A1: LLVMFuncParam, A2: LLVMFuncParam> LLVMFuncParams for (A1, A2) {}
impl<A1: LLVMFuncParam, A2: LLVMFuncParam, A3: LLVMFuncParam> LLVMFuncParams for (A1, A2, A3) {}
impl<A1: LLVMFuncParam, A2: LLVMFuncParam, A3: LLVMFuncParam, A4: LLVMFuncParam> LLVMFuncParams
    for (A1, A2, A3, A4)
{
}

pub trait LLVMFuncResult {}
