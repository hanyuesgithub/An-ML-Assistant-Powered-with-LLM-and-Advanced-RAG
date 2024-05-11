from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from FlagEmbedding import BGEM3FlagModel
import numpy as np

app = FastAPI()

model = BGEM3FlagModel('/root/FlagEmbedding/embedding_model/BAAI_bge-m3', use_fp16=True)

class SentencePair(BaseModel):
    sentence1: str
    sentence2: str
    
class Sentence(BaseModel):
    sentence: str

@app.post("/similarity")
async def calculate_similarity(pair: SentencePair):
    embeddings_1 = model.encode([pair.sentence1], batch_size=12, max_length=8192)['dense_vecs']
    embeddings_2 = model.encode([pair.sentence2])['dense_vecs']
    similarity = np.dot(embeddings_1, embeddings_2.T).item()
    return {"similarity": similarity}

@app.post("/embedding")
async def generate_embedding(sentence: Sentence):
    embedding = model.encode([sentence.sentence], batch_size=12, max_length=8192)['dense_vecs']
    
    return {"embedding": embedding.tolist()}

@app.post("/embedding_test")
async def generate_embedding(sentence: Sentence):
    embedding = model.encode([sentence.sentence], batch_size=12, max_length=8192)['dense_vecs']
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding size: {embedding.size}")
    print(f"Embedding dtype: {embedding.dtype}")
    return {"embedding": embedding.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5116)