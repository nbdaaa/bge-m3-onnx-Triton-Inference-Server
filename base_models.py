from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class EmbeddingRequest(BaseModel):
    input: str 

class EmbeddingData(BaseModel):  # Return as Embedding Response
    dense:  List[float]
    sparse: Dict[str, float]

class BatchEmbeddingRequest(BaseModel):
    input: List[str]
    batch_size: Optional[int] = 16

class BatchEmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
