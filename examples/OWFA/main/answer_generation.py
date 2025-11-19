import time
from icecream import ic
from openai import OpenAI
from utils.config_loader import get_llm_client_config

llm_config = get_llm_client_config()
# =========================
# LLM answer generation 
# =========================
def generate_answer(query, MODEL_NAME, MODEL_CACHE):
    print(f"Generating answer for query: {query}")
    start_time = time.perf_counter()
    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"]
    )
    result = client.chat.completions.create(
        model=llm_config["model_name"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always answer in less than 100 words in a single paragraph without any special formatting."},
            {"role": "user", "content": query}
        ],
        max_tokens=150,
        temperature=0.3,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    )
    answer_text = result.choices[0].message.content
    print(f"Raw answer: {answer_text}")
    if answer_text.startswith(query):
        answer_text = answer_text[len(query):].strip()
    # anything before "</think>", including "</think>" should be removed.
    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>")[-1].strip()
    end_time = time.perf_counter()
    ic(f"generate_answer took {end_time - start_time:.3f} seconds")
    return answer_text
