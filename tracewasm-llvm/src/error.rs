use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("phi instructions should be added at the start of the basic block")]
    PhiInstructionAddError,
}
