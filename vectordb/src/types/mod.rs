pub mod error;
pub mod multimodal;
pub mod record;
pub mod sparse_vector;
pub mod table;
pub mod vector;

pub use error::*;
pub use multimodal::{
    FusionStrategy, ModalityType, MultimodalContent, MultimodalData,
    MultiVectorRecord, CreateMultiVectorRequest, CrossModalSearchQuery,
    MultimodalInput, MultimodalSearchResult, MultimodalSearchHit,
    EmbeddingConfig, EmbeddingRequest, EmbeddingResponse, EmbeddingResult
};
pub use record::{Record, RecordStatus, CreateRecordRequest, SearchQuery, SearchResult, UpdateRecordRequest, SearchHit, Filter};
pub use sparse_vector::*;
pub use table::*;
pub use vector::*;
