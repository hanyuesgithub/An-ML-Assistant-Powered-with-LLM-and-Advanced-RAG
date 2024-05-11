
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# 加载模型和分词器
model_id = "/root/sdc1/Yixuan/llama2-summarization"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
).half()
tokenizer = AutoTokenizer.from_pretrained(model_id)

class QuestionContext(BaseModel):
    question: str
    context: str

@app.post("/generate_answer")
async def generate_answer(input_data: QuestionContext):
    question = input_data.question
    context = input_data.context
    
    prompt = f"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. \
Your task is to generate an answer to the given question. \
And your answer should be based on the provided context only.
<</SYS>>

### Question: {question}
### Context: {context} [/INST]
### Answer:
"""
    print("prompt!!!",prompt)    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # generate output
    outputs = model.generate(input_ids=input_ids, max_new_tokens=128, temperature=0.8)  # max_new_tokens=128, larger max_new_tokens longer inference
    # decode output
    model_output = tokenizer.decode(outputs[0])
    generated_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

    print('\nModel output:\n', model_output)  # decoded_output
    print("___" * 20)
    print('\nGenerated answer:\n', generated_answer)
    print("generated_answer!!!",generated_answer)
    
    return {"answer": generated_answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5115)