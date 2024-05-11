import argparse
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn.functional import cosine_similarity
import uvicorn
import base64
import csv
import datetime

import subprocess
import os
from uuid import uuid4
import wave
import io
import logging
import time
import httpx
import requests
import asyncio
from aiohttp import ClientSession
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pandas as pd
import transformers
import torch

import jieba
import sklearn
import torch
import numpy as np
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

import chromadb

from chromadb.config import Settings
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api import Collection
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"

data = pd.read_json('/root/sdb1/SentenceEmbedding-main/SentenceEmbedding-main/All_data_40000_copy copy 2.json')
print("read data!!!!!")
df_q = pd.DataFrame(columns = ['id','question','answer','reference'])
df_c = pd.DataFrame(columns = ['id','content'])
df_q['id'] = data['id']
df_q['question'] = data['question']

df_c['id'] = data['id']
df_c['content'] = data['context']
df_q_text = df_q
print("df_q_text = df_q!!!!!!!!!")

from rank_bm25 import BM25Okapi

content = [jieba.lcut(context) for context in df_c['content']]
bm25 = BM25Okapi(content)
print("bm25 = BM25Okapi(content)!!!!!!!!!!!")
for index,row in df_q_text.iterrows():
    print(index)
    # decoded_series = series.astype(str).apply(lambda x: x.decode('utf-8'))
    doc_scores = bm25.get_scores(jieba.lcut(row["question"]))
    max_score_idx = doc_scores.argsort()[::-1] + 1
    # print(max_score_idx)
    max_score_idx = max_score_idx.tolist()
    df_q_text.at[index, 'reference'] = [str(x) for x in max_score_idx[:5]]
    # print(row['reference'])
    
    
print("start loading json!!!!!!!!")
df_q_text.to_json('/root/sdb1/SentenceEmbedding-main/SentenceEmbedding-main/All_data_40000_copy copy 2.json', orient='records')

print("df_q_text.to_json!!!!!!!!")
# def split_text(text,chunksize):
#   return [text[i:i+chunksize] for i in range(0,len(text),chunksize)]

contentAll = []
for index,row in df_c.iterrows():
    subcontent = row['content']
#   for chunk_text in split_text(subcontent,300):
    contentAll.append({
        'id':index,
        'content':subcontent
    })
    
question_sentences = [row['question'] for index,row in df_q_text.iterrows()]
content_sentences = [x['content'] for x in contentAll]


def getEmbeddingList(inputList):
    print("getembedding!!!!!!!!")
    output = []
    for index, element in enumerate(inputList):
        print(index)  # 打印当前索引
        embedding = getEmbedding(element)
        output.append(embedding)
    return output

# def getEmbedding(question_sentences, normalize_embeddings=True):
    
#     # print("getting")
#     messages1 = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": question_sentences}
#     ]

#     text1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
#     model_inputs1 = tokenizer([text1], return_tensors="pt").to(device)

#     with torch.no_grad():
#         outputs1 = model(**model_inputs1)
#     embeddings1 = outputs1.last_hidden_state.mean(dim=1)

#     return embeddings1


def getEmbedding(sentence):
    url = "http://0.0.0.0:5114/embedding"
    payload = {"sentence": sentence}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        embedding = np.array(response.json()["embedding"])
        return embedding
    else:
        raise Exception(f"请求失败,状态码: {response.status_code}")


print("start getEmbeddingList")

question_embeddings = getEmbeddingList(question_sentences)
pdf_embeddings = getEmbeddingList(content_sentences)
df_q_text_ = df_q_text
def remove_duplicates(input_list):
  seen =set()
  result = []
  for item in input_list:
    if item not in seen:
      seen.add(item)
      result.append(item)
    #   print("item",item)
  return result



# question_embeddings = [feat.cpu().numpy() for feat in question_embeddings]
question_embeddings = [feat for feat in question_embeddings]
# pdf_embeddings = [feat.cpu().numpy() for feat in pdf_embeddings]
pdf_embeddings = [feat for feat in pdf_embeddings]

question_embeddings = np.array(question_embeddings)
pdf_embeddings = np.array(pdf_embeddings)
pdf_embeddings = pdf_embeddings.squeeze(axis=1)
question_embeddings =  question_embeddings.squeeze(axis=1)

ids_question = [str(index) for index in range(len(question_sentences))]
ids_pdf = [str(index) for index in range(len(content_sentences))]



chroma_client = chromadb.PersistentClient(path="chroma_output_new_final_standard")
collection = chroma_client.get_or_create_collection(name="question_embedding")
question_embeddings = [[float(elem) for elem in row] for row in question_embeddings]
collection.add(
    embeddings=question_embeddings,
    documents=question_sentences,
    ids=ids_question
)

# chroma_client2 = chromadb.PersistentClient(path="chroma_output_question")
# 创建或加载集合
collection2 = chroma_client.get_or_create_collection(name="pdf_embedding")

pdf_embeddings = [[float(elem) for elem in row] for row in pdf_embeddings]

print(len(pdf_embeddings))

collection2.add(
    embeddings=pdf_embeddings,
    documents=content_sentences,
    ids=ids_pdf
)



# query = "What is the first document about?"
# query_embedding = getEmbedding(query)
# # query_embedding = query_embedding.tolist()  # 将 PyTorch 张量转换为列表
# query_embedding = query_embedding.cpu().numpy().tolist()
# print("query_embedding!!!!!!!!!")
# print(query_embedding)
# query_embedding = [[float(elem) for elem in row] for row in query_embedding]

# # print(query_embedding.shape)
# # print(query_embedding)


# results = collection.query(
#     query_embeddings=query_embedding,
#     n_results=3  # 检索的文档数量
# )
# print("result!!!!!!!!",results)

# # 提取检索到的文档
# # retrieved_docs = [doc["text"] for doc in results["documents"][0]]
# # print("retrieved_docs",retrieved_docs)
# retrieved_docs = results["documents"][0]
# print("retrieved_docs",retrieved_docs)

