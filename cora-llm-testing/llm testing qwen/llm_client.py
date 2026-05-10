import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"
#MODEL = "deepseek-coder:6.7b"

def call_llm(messages):

    prompt = messages[-1]["content"]
    prompt = prompt[-6000:]   

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1   
            },
            timeout=600
        )

        r.raise_for_status()
        return r.json().get("response", "")

    except Exception as e:
        return f"[LLM_ERROR] {str(e)}"
