from fastapi import FastAPI
import chromadb
import numpy as np
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import jieba
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def get_context_collection():
    
    chroma_client = chromadb.PersistentClient(path="/root/sdb1/SentenceEmbedding-main/chroma_output_new_final")
    # chroma_client = chromadb.PersistentClient(path="/root/sdb1/SentenceEmbedding-main/chroma_output_new_final_standard")
    collection = chroma_client.get_or_create_collection(name="question_embedding")
    collection2 = chroma_client.get_or_create_collection(name="pdf_embedding")
    return collection, collection2

def getEmbedding(sentence):
    url = "http://0.0.0.0:5116/embedding"
    payload = {"sentence": sentence}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        embedding = np.array(response.json()["embedding"])
        return embedding
    else:
        raise Exception(f"请求失败,状态码: {response.status_code}")

def get_context(input, rerank_model):
    collection_q, collection_c = get_context_collection()

    question = collection_q.get()
    df_q = pd.DataFrame(question)
    df_q['ids'] = df_q['ids'].apply(lambda x: int(x))
    df_q = df_q.sort_values(by='ids')
    df_q.reset_index(drop=True, inplace=True)

    content = [jieba.lcut(context) for context in df_q['documents'].tolist()]
    bm25 = BM25Okapi(content)
    doc_scores = bm25.get_scores(jieba.lcut(input))
    bm25_index = doc_scores.argsort()[::-1][:3].astype(str)

    input_embedding = getEmbedding(input)
    embedding_index = collection_q.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    fusion_result = []
    k = 60
    fusion_score = {}
    for idx, q in enumerate(bm25_index):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(embedding_index):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)
    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    pairs = []
    for sorted_result in sorted_dict[:5]:
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        pairs.append([input, context])
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    sorted_result = sorted_dict[scores.cpu().numpy().argmax()]
    match_context = collection_c.get(ids=[sorted_result[0]])['documents'][0]

    return match_context


def get_context_q_to_c(input, rerank_model):
    collection_q, collection_c = get_context_collection()
    context = collection_c.get()
    df_c = pd.DataFrame(context)
    df_c['ids'] = df_c['ids'].apply(lambda x: int(x))
    df_c = df_c.sort_values(by='ids')
    df_c.reset_index(drop=True, inplace=True)
    
    content = [jieba.lcut(context) for context in df_c['documents'].tolist()]
    bm25 = BM25Okapi(content)
    doc_scores = bm25.get_scores(jieba.lcut(input))
    bm25_index = doc_scores.argsort()[::-1][:3].astype(str)

    input_embedding = getEmbedding(input)
    embedding_index = collection_c.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    fusion_result = []
    k = 60
    fusion_score = {}
    for idx, c in enumerate(bm25_index):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)

    for idx, c in enumerate(embedding_index):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)
    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    pairs = []
    for sorted_result in sorted_dict[:5]:
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        pairs.append([input, context])
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    sorted_result = sorted_dict[scores.cpu().numpy().argmax()]
    match_context = collection_c.get(ids=[sorted_result[0]])['documents'][0]

    return match_context

tokenizer = AutoTokenizer.from_pretrained("/root/sdc1/Yixuan/QA-ml-arxiv-papers-main/embedding/bge-reranker-large/quietnight/bge-reranker-large")
rerank_model = AutoModelForSequenceClassification.from_pretrained("/root/sdc1/Yixuan/QA-ml-arxiv-papers-main/embedding/bge-reranker-large/quietnight/bge-reranker-large")
rerank_model.cuda()
rerank_model.eval()



def get_context_embedding_only(input, top_k=3):
    collection_q, collection_c = get_context_collection()

    question = collection_q.get()
    df_q = pd.DataFrame(question)
    df_q['ids'] = df_q['ids'].apply(lambda x: int(x))
    df_q = df_q.sort_values(by='ids')
    df_q.reset_index(drop=True, inplace=True)

    content = [jieba.lcut(context) for context in df_q['documents'].tolist()]
    bm25 = BM25Okapi(content)
    doc_scores = bm25.get_scores(jieba.lcut(input))
    bm25_index = doc_scores.argsort()[::-1][:3].astype(str)

    input_embedding = getEmbedding(input)
    embedding_index = collection_q.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    fusion_result = []
    k = 60
    fusion_score = {}
    for idx, q in enumerate(bm25_index):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(embedding_index):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)
    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    match_contexts = []
    for sorted_result in sorted_dict[:top_k]:
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        match_contexts.append(context)

    return match_contexts



def get_context_embedding_q_t_c(input, top_k=3):
    collection_q, collection_c = get_context_collection()
    context = collection_c.get()
    df_c = pd.DataFrame(context)
    df_c['ids'] = df_c['ids'].apply(lambda x: int(x))
    df_c = df_c.sort_values(by='ids')
    df_c.reset_index(drop=True, inplace=True)
    
    content = [jieba.lcut(context) for context in df_c['documents'].tolist()]
    bm25 = BM25Okapi(content)
    doc_scores = bm25.get_scores(jieba.lcut(input))
    bm25_index = doc_scores.argsort()[::-1][:3].astype(str)

    input_embedding = getEmbedding(input)
    embedding_index = collection_c.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    fusion_result = []
    k = 60
    fusion_score = {}
    for idx, c in enumerate(bm25_index):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)

    for idx, c in enumerate(embedding_index):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)
    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    match_contexts = []
    for sorted_result in sorted_dict[:top_k]:
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        match_contexts.append(context)

    return match_contexts

def get_context_combined(input, rerank_model, top_k=3):
    collection_q, collection_c = get_context_collection()

    question = collection_q.get()
    df_q = pd.DataFrame(question)
    df_q['ids'] = df_q['ids'].apply(lambda x: int(x))
    df_q = df_q.sort_values(by='ids')
    df_q.reset_index(drop=True, inplace=True)

    content_q = [jieba.lcut(context) for context in df_q['documents'].tolist()]
    bm25_q = BM25Okapi(content_q)
    doc_scores_q = bm25_q.get_scores(jieba.lcut(input))
    bm25_index_q = doc_scores_q.argsort()[::-1][:3].astype(str)

    input_embedding = getEmbedding(input)
    embedding_index_q = collection_q.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    context = collection_c.get()
    df_c = pd.DataFrame(context)
    df_c['ids'] = df_c['ids'].apply(lambda x: int(x))
    df_c = df_c.sort_values(by='ids')
    df_c.reset_index(drop=True, inplace=True)
    
    content_c = [jieba.lcut(context) for context in df_c['documents'].tolist()]
    bm25_c = BM25Okapi(content_c)
    doc_scores_c = bm25_c.get_scores(jieba.lcut(input))
    bm25_index_c = doc_scores_c.argsort()[::-1][:3].astype(str)

    embedding_index_c = collection_c.query(
        query_embeddings=input_embedding,
        n_results=6)['ids'][0]

    fusion_result = []
    k = 60
    fusion_score = {}
    for idx, q in enumerate(bm25_index_q):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(embedding_index_q):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, c in enumerate(bm25_index_c):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)

    for idx, c in enumerate(embedding_index_c):
        if c not in fusion_score:
            fusion_score[c] = 1 / (idx + k)
        else:
            fusion_score[c] += 1 / (idx + k)

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    pairs = []
    for sorted_result in sorted_dict[:5]:
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        pairs.append([input, context])
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    match_contexts = []
    for i in range(top_k):
        sorted_result = sorted_dict[scores.cpu().numpy().argsort()[::-1][i]]
        id = sorted_result[0]
        context = collection_c.get(ids=[id])['documents'][0]
        match_contexts.append(context)

    return match_contexts

class Request(BaseModel):
    input: str
    
@app.post("/get_context")
async def get_context_api(input_data: Request):
    input = input_data.input
    context = get_context(input, rerank_model)
    return {"context": context}

@app.post("/get_context_q_to_c")
async def get_context_api(input_data: Request):
    input = input_data.input
    context = get_context_q_to_c(input, rerank_model)
    return {"context": context}

@app.post("/get_context_embedding_only")
async def get_context_api(input_data: Request):
    input = input_data.input
    resultArr = get_context_embedding_only(input, top_k=3)
    return {"resultArr": resultArr}

@app.post("/get_context_embedding_q_t_c")
async def get_context_api(input_data: Request):
    input = input_data.input
    resultArr = get_context_embedding_q_t_c(input, top_k=3)
    print(resultArr)
    return {"resultArr": resultArr}

@app.post("/get_context_combined")
async def get_context_api(input_data: Request):
    input = input_data.input
    resultArr = get_context_combined(input, rerank_model, top_k=3)
    return {"resultArr": resultArr}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5117)
