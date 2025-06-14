from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import re, os, uuid, json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import httpx

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

app = FastAPI(
    title="Precise Scraper + Vector + LLM",
    description="Scrapes, merges short lines with context, stores clean vectors, answers via Groq LLaMA.",
    version="3.6"
)

TAGS_TO_EXTRACT = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "span", "b", "strong", "div"]
VECTOR_STORE_PATH = "./vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
GROQ_API_KEY = "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"  # ← Replace this

# ------------------------- Scraping & Merging -------------------------

def extract_visible_content(driver) -> List[str]:
    script = f"""
        const tags = {TAGS_TO_EXTRACT};
        const content = [];
        tags.forEach(tag => {{
            const elements = document.querySelectorAll(tag);
            elements.forEach(el => {{
                const style = window.getComputedStyle(el);
                if (
                    el.innerText &&
                    el.innerText.trim().length > 0 &&
                    style.visibility !== 'hidden' &&
                    style.display !== 'none'
                ) {{
                    content.push(el.innerText.trim());
                }}
            }});
        }});
        return content;
    """
    raw = driver.execute_script(script)
    return [re.sub(r'\s+', ' ', item).strip() for item in raw if item.strip()]

def smart_merge_lines(lines: List[str], max_len: int = 400) -> List[str]:
    """Merge short lines with context — e.g., 'Code:' + 'GODARSHAN'."""
    merged = []
    buffer = ""
    for i, line in enumerate(lines):
        if len(line) < 40 or (buffer and len(buffer) + len(line) < max_len):
            buffer += " " + line
        else:
            if buffer.strip():
                merged.append(buffer.strip())
            buffer = line
    if buffer.strip():
        merged.append(buffer.strip())

    # Final dedup
    seen = set()
    clean = []
    for para in merged:
        para = re.sub(r'\s+', ' ', para).strip()
        if para and para not in seen and len(para) > 10:
            seen.add(para)
            clean.append(para)
    return clean

def scrape_and_extract(url: str) -> List[str]:
    options = Options()
    options.headless = False
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = uc.Chrome(options=options)
    wait = WebDriverWait(driver, 10)
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    lines = extract_visible_content(driver)

    # Try clicking all tabs
    tab_buttons = driver.find_elements(By.XPATH, "//button")
    clicked = set()
    for btn in tab_buttons:
        try:
            label = btn.text.strip()
            if not label or label in clicked:
                continue
            btn.click()
            clicked.add(label)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            lines += extract_visible_content(driver)
        except Exception:
            continue

    driver.quit()
    return smart_merge_lines(lines)

# ------------------------- API Endpoints -------------------------

@app.post("/scrape_store")
def scrape_store(data: Dict[str, Any]):
    url = data.get("url")
    if not url:
        return JSONResponse(content={"error": "URL is required"}, status_code=400)

    try:
        paragraphs = scrape_and_extract(url)
        vectors = embedding_model.encode(paragraphs).tolist()
        search_code = str(uuid.uuid4())[:8]
        with open(f"{VECTOR_STORE_PATH}/{search_code}.json", "w") as f:
            json.dump({"texts": paragraphs, "vectors": vectors}, f)
        return {"success": True, "search_code": search_code}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.post("/search_llm")
async def search_llm(data: Dict[str, Any]):
    query = data.get("query")
    code = data.get("search_code")
    if not query or not code:
        return JSONResponse(content={"error": "query and search_code are required"}, status_code=400)

    try:
        file_path = f"{VECTOR_STORE_PATH}/{code}.json"
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": "Invalid search_code"}, status_code=404)

        with open(file_path, "r") as f:
            stored = json.load(f)

        query_vec = embedding_model.encode([query])
        stored_vecs = np.array(stored["vectors"])
        sims = cosine_similarity(query_vec, stored_vecs)[0]

        top_chunks = sorted(
            [(score, stored["texts"][idx]) for idx, score in enumerate(sims)],
            reverse=True
        )
        top_chunks = [(s, t) for s, t in top_chunks if s > 0.35][:5]

        if not top_chunks:
            return {"response": "No relevant information found to answer this."}

        context = "\n\n".join(list(dict.fromkeys([t for _, t in top_chunks])))
        best_score = top_chunks[0][0]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions based only on the provided content. Keep answers natural and don't make it look like you are referring to some content. Only answer if content is relevant to the asked question otherwise say: Sorry, i don't think i can answer this :)"
            },
            {
                "role": "user",
                "content": f"Content:\n{context}\n\nAnswer this question: {query}"
            }
        ]

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300
        }

        async with httpx.AsyncClient() as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            result = res.json()

        return {
            "matched_score": round(best_score, 2),
            "matched_chunks": [t for _, t in top_chunks],
            "response": result["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 10000))  # Render will inject PORT; fallback to 10000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

