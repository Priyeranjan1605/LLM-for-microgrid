import streamlit as st
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
@st.cache_data
def load_dataset(jsonl_file='microgrid_dataset_expanded.jsonl'):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    prompts = [entry['prompt'] for entry in data]
    answers = [entry['completion'] for entry in data]
    return prompts, answers

prompts, answers = load_dataset()
@st.cache_resource
def build_faiss_index(prompts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompts, convert_to_tensor=True)
    embeddings_np = np.array(embeddings.cpu(), dtype='float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, model

index, model = build_faiss_index(prompts)
def search_similar(query, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype='float32'), top_k)
    results = [{"prompt": prompts[i], "answer": answers[i]} for i in I[0]]
    return results
def safety_check(battery_soc, load_kw, grid_freq, answer):
    warnings = []
    if battery_soc < 20:
        warnings.append("⚠ Battery SOC too low! Do not discharge below 20%.")
    if load_kw > 10:
        warnings.append(f"⚠ Load ({load_kw} kW) may exceed inverter capacity!")
    if grid_freq < 49 or grid_freq > 51:
        warnings.append(f"⚠ Grid frequency abnormal: {grid_freq:.2f} Hz.")
    if answer.strip() == "":
        warnings.append("⚠ Assistant could not provide a safe recommendation.")
    return warnings
def get_live_data():
    battery_soc = np.random.randint(25, 80)
    load_kw = np.random.uniform(2, 8)
    solar_output = np.random.uniform(0, 5)
    grid_freq = np.random.uniform(49.5, 50.5)
    temperature = np.random.uniform(20, 40)
    return battery_soc, load_kw, solar_output, grid_freq, temperature
st.title("Microgrid LLM Assistant (Live Dashboard)")
st.markdown("Live microgrid readings update every 5 seconds with LLaMA 2 recommendations.")

query = st.text_area("Enter your question:", value="How should I manage battery and solar output?")
live_data_area = st.empty()
answer_area = st.empty()
warnings_area = st.empty()
while True:
    battery_soc, load_kw, solar_output, grid_freq, temperature = get_live_data()
    
    live_data_area.markdown(
        f"**Live Data:**\n"
        f"- Battery SOC: {battery_soc}%\n"
        f"- Load: {load_kw:.2f} kW\n"
        f"- Solar Output: {solar_output:.2f} kW\n"
        f"- Grid Frequency: {grid_freq:.2f} Hz\n"
        f"- Temperature: {temperature:.1f} °C"
    )
    
    sensor_context = (
        f"Battery SOC: {battery_soc}%\n"
        f"Load: {load_kw:.2f} kW\n"
        f"Solar Output: {solar_output:.2f} kW\n"
        f"Grid Frequency: {grid_freq:.2f} Hz\n"
        f"Temperature: {temperature:.1f} °C"
    )
    
    results = search_similar(query, top_k=3)
    context = "\n".join([f"Q: {r['prompt']}\nA: {r['answer']}" for r in results])
    payload = {
        "model": "llama2",
        "messages": [
            {"role": "user", "content": context + "\n" + sensor_context + "\nQ: " + query + "\nA:"}
        ]
    }
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
    warnings = safety_check(battery_soc, load_kw, grid_freq, answer)
    answer_area.text_area("Assistant's Answer:", value=answer, height=200)
    if warnings:
        warnings_text = "\n".join(warnings)
        warnings_area.warning(warnings_text)
    else:
        warnings_area.empty()
    time.sleep(5)
