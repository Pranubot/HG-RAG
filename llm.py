"""
llm.py is a thin wrapper around a locally running Ollama instance.

Default model is "Mistral"; I chose this model due to its recency, high parameter count of 7 billion
and high performance given the smaller size. I also desired a model that wasn't overengineered for our
task or too large for the average enthusiast to download.
"""

import requests

OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
EMBED_URL    = "http://127.0.0.1:11434/api/embeddings"
DEFAULT_MODEL = "mistral:latest"
EMBED_MODEL   = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text


def query_llm(prompt, model=DEFAULT_MODEL, system=None, max_tokens=400):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180,
                             proxies={"http": None, "https": None})
        if resp.status_code != 200:
            return f"[LLM ERROR: HTTP {resp.status_code} — {resp.text[:300]}]"
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[LLM ERROR: Ollama not running. Start with: ollama serve]"
    except Exception as e:
        return f"[LLM ERROR: {e}]"


def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    try:
        resp = requests.post(
            EMBED_URL,
            json={"model": model, "prompt": text},
            timeout=60,
            proxies={"http": None, "https": None},
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("embedding", [])
    except Exception:
        return []


def check_ollama():
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# Prompts

HGRAG_SYSTEM = (
    "You are a world knowledge assistant. "
    "Answer questions using ONLY the structured context provided. "
    "Do not add any information not present in the context. "
    "If the answer is not in the context, say 'Not found in context.' "
    "For trade and supply questions, only consider cities connected by trade_with relations — not borders. "
    "Be concise: 1-2 sentences for simple facts, up to 3-4 sentences when naming multiple entities or explaining a causal chain."
)

BASELINE_SYSTEM = (
    "You are a world knowledge assistant. "
    "Answer questions using the world information provided. "
    "Be concise and factual: 1-2 sentences for simple facts, up to 3-4 sentences when naming multiple entities or explaining a causal chain."
)


def build_hgrag_prompt(context, query):
    prompt = (
        f"Context:\n{context}\n"
        f"[The [LOCATION CHAIN] section above shows the containment hierarchy "
        f"(e.g. city → country → planet). Use it to answer location and hierarchy questions.]\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return prompt, HGRAG_SYSTEM


def build_baseline_prompt(context, query):
    prompt = f"World Information:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return prompt, BASELINE_SYSTEM


SIMPLE_RAG_SYSTEM = (
    "You are a world knowledge assistant. "
    "Answer questions using ONLY the retrieved context passages provided. "
    "Do not add any information not present in the context. "
    "If the answer is not in the context, say 'Not found in context.' "
    "Be concise: 1-2 sentences for simple facts, up to 3-4 sentences when naming multiple entities or explaining a causal chain."
)


def build_rag_prompt(context: str, query: str):
    prompt = f"Retrieved Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return prompt, SIMPLE_RAG_SYSTEM

"""
Pranu's note: Judging Multi-Hop questions were a unique challenge, large part due to them being open-ended questions. Not only did I want
the correct answer (which was a simple keyword lookup), I also wanted a correct explanation to demonstrate inference from the LLM. This
wasn't easily done with keywords, so I employed the use of the mistral LLM, but as a judge this time. The rubric reflects an conscious 
decision I made about which part of the answer had more importance, the entity or the explanation. LLms are able to discern the reasoning
behind the answer for a question they are unable to answer purely on context clues. So while the baseline LLM may hallucinate the entity,
they can give a proper reasoning due to it not being directly tied to the graph. Therefore, getting the entity or entities correct has a
larger impact on the 1-5 grade the LLM judge will give. (Look at rubric below)

"""


JUDGE_SYSTEM = (
    "You are evaluating whether an AI correctly identified the specific entities "
    "(cities or countries) involved in a virtual world scenario.\n\n"
    "Evaluation procedure:\n"
    "  1. Read the Expected Answer — it names the correct entities explicitly.\n"
    "  2. Check whether those exact entity names appear in the AI Response.\n"
    "  3. Penalize heavily for entities in the AI Response that are NOT in the Expected Answer, "
    "     even if the reasoning sounds plausible. A confident, well-structured answer that names "
    "     wrong entities is fabrication and cannot score above 2.\n\n"
    "Score on a scale of 1 to 5:\n"
    "  5 — All correct entities present, correct reasoning\n"
    "  4 — All correct entities present, reasoning mostly sound with minor gaps\n"
    "  3 — Some correct entities, some wrong, reasoning sound\n"
    "  2 — Wrong entities but the reasoning approach is valid\n"
    "  1 — Wrong entities and no valid reasoning, or response is a fabricated list\n\n"
    "Respond in exactly this format, nothing else:\n"
    "SCORE: <integer 1-5>\n"
    "REASON: <one concise sentence>"
)


ENTITY_EXTRACTION_SYSTEM = (
    "You are an entity extractor for a fictional world knowledge graph. "
    "Given a question, extract the specific proper noun (a city, country, or planet name) that is the "
    "SUBJECT of the question — the named entity you would need to look up to answer it, "
    "NOT the type of answer being requested. "
    "For example: 'What country is Ashburg in?' → extract 'Ashburg', not 'country'. "
    "'Which planet does Barnes belong to?' → extract 'Barnes', not 'planet'. "
    "Reply with only the entity name, nothing else — no explanation, no punctuation."
)


def build_entity_extraction_prompt(query: str):
    return query, ENTITY_EXTRACTION_SYSTEM


def build_judge_prompt(query: str, answer_key: str, response: str):
    prompt = (
        f"Question: {query}\n\n"
        f"Expected Answer: {answer_key}\n\n"
        f"AI Response: {response}"
    )
    return prompt, JUDGE_SYSTEM
