from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import re, os, uuid, json, asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx
from playwright.async_api import async_playwright

app = FastAPI()

TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "span", "b", "strong", "div"]
VECTOR_STORE_PATH = "./vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

embedding_model = None  # Will be loaded later
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# --------------------- Scraper with Playwright ---------------------

async def extract_visible_content(page) -> List[str]:
    script = f"""
        const tags = {TAGS};
        const content = [];
        tags.forEach(tag => {{
            document.querySelectorAll(tag).forEach(el => {{
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
    return await page.evaluate(script)

async def scrape_with_playwright(url: str) -> List[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            "--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"
        ])
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        content = await extract_visible_content(page)

        buttons = await page.query_selector_all("button")
        for btn in buttons:
            try:
                label = await btn.inner_text()
                if not label.strip():
                    continue
                await btn.click()
                await page.wait_for_timeout(1000)
                content += await extract_visible_content(page)
            except:
                continue

        await page.close()
        await browser.close()
        return merge_paragraphs(content)

def merge_paragraphs(lines: List[str], max_len=400) -> List[str]:
    merged = []
    buf = ""
    for line in lines:
        if len(line) < 40 or (buf and len(buf) + len(line) < max_len):
            buf += " " + line
        else:
            if buf.strip():
                merged.append(buf.strip())
            buf = line
    if buf.strip():
        merged.append(buf.strip())

    seen, final = set(), []
    for para in merged:
        clean = re.sub(r'\s+', ' ', para).strip()
        if clean and clean not in seen and len(clean) > 10:
            seen.add(clean)
            final.append(clean)
    return final

# --------------------- API Endpoints ---------------------

@app.on_event("startup")
async def load_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

@app.post("/scrape_store")
def scrape_store(data: Dict[str, Any]):
    url = data.get("url")
    if not url:
        return JSONResponse(content={"error": "Missing URL"}, status_code=400)

    try:
        paragraphs = asyncio.run(scrape_with_playwright(url))
        vectors = embedding_model.encode(paragraphs).tolist()

        code = str(uuid.uuid4())[:8]
        with open(f"{VECTOR_STORE_PATH}/{code}.json", "w") as f:
            json.dump({"texts": paragraphs, "vectors": vectors}, f)

        return {"success": True, "search_code": code}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post("/search_llm")
async def search_llm(data: Dict[str, Any]):
    query = data.get("query")
    code = data.get("search_code")
    if not query or not code:
        return JSONResponse(content={"error": "query and search_code required"}, status_code=400)

    try:
        path = f"{VECTOR_STORE_PATH}/{code}.json"
        if not os.path.exists(path):
            return JSONResponse(content={"error": "Invalid code"}, status_code=404)

        with open(path, "r") as f:
            store = json.load(f)

        qvec = embedding_model.encode([query])
        svecs = np.array(store["vectors"])
        sims = cosine_similarity(qvec, svecs)[0]

        top = sorted([(s, store["texts"][i]) for i, s in enumerate(sims)], reverse=True)
        top = [(s, t) for s, t in top if s > 0.35][:5]

        if not top:
            return {"response": "No relevant information found."}

        context = "\n\n".join(dict.fromkeys([t for _, t in top]))
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
            "matched_score": round(top[0][0], 2),
            "matched_chunks": [t for _, t in top],
            "response": result["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional: local run support
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
