import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
with open('microgrid_dataset_expanded.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
prompts = [entry["prompt"] for entry in data]
answers = [entry["completion"] for entry in data]
print("Creating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(prompts, convert_to_tensor=True)
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings.cpu()))

def search_similar(query, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    results = []
    for i in I[0]:
        results.append({"prompt": prompts[i], "answer": answers[i]})
    return results
print("Microgrid Assistant ready! Type 'exit' to quit.")

while True:
    query = input("\nEnter your question: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    results = search_similar(query, top_k=3)
    context = "\n".join([f"Q: {r['prompt']}\nA: {r['answer']}" for r in results])
    payload = {
        "model": "llama2",
        "messages": [
            {"role": "user", "content": context + "\nQ: " + query + "\nA:"}
        ]
    }
    print("Sending request to Ollama...")
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    message_content = data.get('message', {}).get('content', "")
                    answer += message_content
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException as e:
        answer = f"Error communicating with Ollama: {e}"
    print("\nAssistant's Answer:")
    print(answer)