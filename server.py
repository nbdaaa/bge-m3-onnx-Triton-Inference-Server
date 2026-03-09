from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from base_models import EmbeddingRequest, EmbeddingData, BatchEmbeddingRequest, BatchEmbeddingResponse
import os, logging

from triton_client import TritonBGEM3Client  # thay OnnxQueryEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

_embedding_model: TritonBGEM3Client = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Triton Inference Server...")
    get_embedding_model()
    logger.info("Triton client ready")
    yield


def get_embedding_model() -> TritonBGEM3Client:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TritonBGEM3Client(
            triton_url=os.getenv("TRITON_URL", "localhost:8000"),
            max_length=int(os.getenv("MAX_LENGTH", "512")),
        )
    return _embedding_model


app = FastAPI(title="Document Embedding API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/encode", response_model=EmbeddingData)
async def encode(request: EmbeddingRequest):
    logger.info(f"Received encode request for text length: {len(request.input)}")
    model = get_embedding_model()
    vectors = model.encode(str(request.input))
    logger.info("Encode request completed")

    return EmbeddingData(
        dense=vectors["dense"],
        sparse=vectors["sparse"]
    )


@app.post("/encode_batch", response_model=BatchEmbeddingResponse)
async def encode_batch(request: BatchEmbeddingRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")

    logger.info(f"Received batch encode request for {len(request.input)} texts with batch_size={request.batch_size}")
    model = get_embedding_model()
    batch_size = request.batch_size or 16
    results = model.encode_batch(request.input, batch_size=batch_size)
    logger.info(f"Batch encode request completed, processed {len(results)} texts")

    return BatchEmbeddingResponse(
        data=[EmbeddingData(dense=r["dense"], sparse=r["sparse"]) for r in results]
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345)