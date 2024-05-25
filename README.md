Hi there!! Let me introduce our ML assistant built in this project! 😊 

![UI](https://github.com/hanyuesgithub/An-ML-Assistant-Powered-with-LLM-and-Advanced-RAG/assets/80732009/ae5f2068-e63a-41f3-8975-0df0a28d06ae)

We developed an intelligent system powered with large language models(hashtag#LLMs) and advanced Retrieval-Augmented Generation(hashtag#RAG) supported with over 100,000 machine learning papers, aiming to provide ML learners and researchers with a user-friendly platform for easily understanding the intricate principles and methodologies behind burgeoning ML concepts.

The system flow includes three pivotal components: 
a) Data Vectorization: ML papers vectorized as embedding representations and stored in database;
b) Context Retrieval: User queries to match with ML papers chunks which return as context;
c) LLM Inference: LLM finetuned with ML paper data to generate answers to AI-related user queries. 
![system_design](https://github.com/hanyuesgithub/An-ML-Assistant-Powered-with-LLM-and-Advanced-RAG/assets/80732009/96d7947d-5a9f-4bb4-bb2b-e59ef54d72bb)

So, what has been done to build the system?
✅ Created a question-answering dataset (ml-arxiv-papers-qa)
✅ Finetuned LLMs (hashtag#Llama-2-7B-Chat and hashtag#Mistral-7B)
✅ Trained an embedding model (BGE-M3)
✅ Built a RAG system with combined retrieval methods (BM25, BGE-M3 and Reranking)
✅ Designed and created a conversational UI

Our finetuned Llama-2-7B-Chat achieves statistically significant improvements and it shows the strongest performance across all evaluation metrics we applied compared to the base Llama-2-7B-Chat and base Llama-3-8B-Instruct models. Besides, our RAG system achieves a retrieval accuracy of 99.75%!
![llama_performance_comparison](https://github.com/hanyuesgithub/An-ML-Assistant-Powered-with-LLM-and-Advanced-RAG/assets/80732009/159b03c5-bf27-4412-8ad7-ccb5fdde7bfc)


Teamwork divides the task and multiplies the success! Deeply appreciate the active engagement and collaborative spirit of the team (Hanyue Liu, Kenan Zhang, Linze Li, Xinyu Wang, Yixuan Gong)! 🎉

PS: the question-answering dataset and the finetuned Llama-2-7B-Chat checkpoint are publicly available on Huggingface:  
https://huggingface.co/datasets/hanyueshf/ml-arxiv-papers-qa; https://huggingface.co/hanyueshf/llama-2-7b-chat-ml-qa 

