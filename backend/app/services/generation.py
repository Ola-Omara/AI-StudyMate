import re

import ollama

from app.core.config import Settings

REFUSAL_TEXT = "I don't have enough information in my verified sources to answer this question."
OUT_OF_SCOPE_TEXT = "I only answer questions about Artificial Intelligence, Machine Learning, and Deep Learning."

CHITCHAT_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|yo)\s*[!.]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*good\s+(morning|afternoon|evening)\s*[!.]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+are\s+you\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*thanks?(\s+you)?\s*[!.]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|see\s+you)\s*[!.]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*who\s+are\s+you\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+can\s+you\s+help\s+me\s+with\??\s*$", re.IGNORECASE),
]

CHITCHAT_RESPONSES = {
    "greeting": "Hello! I'm AI StudyMate. Ask me anything about Machine Learning or Deep Learning.",
    "how_are_you": "I'm doing well, thanks for asking! What ML or DL topic can I help you with?",
    "thanks": "You're welcome! Let me know if you have another ML or DL question.",
    "farewell": "Goodbye! Come back anytime you have an ML or DL question.",
    "who_are_you": "I'm AI StudyMate, a RAG assistant that answers Machine Learning and Deep Learning questions using a verified academic corpus.",
    "what_can_you_help_with": "I can answer questions about Machine Learning and Deep Learning topics such as regression, classification, clustering, neural networks, CNNs, RNNs, and more, grounded in a verified set of academic sources.",
}

INTENT_CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a study assistant that only answers questions about Artificial Intelligence, Machine Learning, and Deep Learning: supervised/unsupervised learning, regression, classification, clustering, decision trees, SVM, k-NN, Naive Bayes, ensemble methods, model evaluation, overfitting/underfitting, feature selection, neural networks, backpropagation, CNNs, RNNs, LSTMs, autoencoders, optimization, regularization, attention, Transformers, GANs, VAEs, diffusion models.

Classify the user's message into exactly one label:
IN_SCOPE - a genuine question about one of the AI/ML/DL topics above.
OUT_OF_SCOPE - about anything else (cooking, sports, politics, travel, movies, general programming unrelated to AI/ML/DL, etc).

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE. Do not explain."""

STRICT_SYSTEM_PROMPT = """You are AI StudyMate, an assistant that answers Machine Learning and Deep Learning questions using ONLY the numbered sources the user provides in their message.

Rules:
1. Use ONLY the information in the numbered sources below the question. Do not use any outside knowledge.
2. Every factual claim must be supported by one of the numbered sources. Cite it using its bracket ID exactly as given to you, for example [S1] or [S2]. Never invent an ID, and never cite an ID that was not given to you.
3. Do not use any other citation format. Do not write author names, page numbers, titles, or section names yourself, anywhere in your answer -- only the bracket ID. Writing "(Random Forests, page 3, Section: X)" yourself is WRONG; write [S1] instead and let the system render it.
4. If a numbered source contains a partial or indirect explanation of the mechanism being asked about, use it and cite it rather than refusing. Only refuse if none of the numbered sources contain information relevant to the question.
5. If, after applying rule 4, the sources still do not support an answer, your ENTIRE response must be nothing but the exact text: AI_STUDYMATE_REFUSAL
   Do not write any analysis, partial reasoning, or citations before or after it. Do not explain why you are refusing. If you catch yourself starting to reason about which source is missing what, stop and output only the refusal string instead.
6. ALWAYS write your answer as 2-5 full, explanatory sentences in your own words, with bracket citations woven in. A response that is only citation tags with no explanatory sentence (for example: "[S1] [S2]" or "(some title) (another title)") is INVALID -- it is not an acceptable answer, and not a valid refusal either. Explain the actual mechanism being asked about.
7. Keep answers concise and direct. Cite only the specific source IDs that support each claim, not every source you were given. Do not greet the user or ask for clarification.
8. Don't say "I don't know" or "I can't answer that" -- if the sources don't support an answer, output only the refusal string.
9. Don't say the refusal string if the sources do support an answer, even if it's partial or indirect. Only use the refusal string if none of the sources contain relevant information.
Example of a GOOD answer, given sources [S1] and [S2]:
"Dropout regularizes a network by randomly zeroing a fraction of hidden units on each training pass, which prevents units from co-adapting to fix each other's mistakes [S1]. At test time all units are kept but their outputs are scaled down to match the expected value seen during training [S2]."

Example of a BAD answer (do not do this -- no explanation, just tags):
"[S1] [S2] [page 3, Section: X] (Random Forests, page 3, Section: X) [S1]"
10.DONOT MENTION OR REPEAT THE SOURCES' TITLES, PAGES, OR SECTIONS IN YOUR ANSWER. or any thing
"""

CITATION_ID_PATTERN = re.compile(r"\[S(\d+)\]")
REFUSAL_SENTINEL = "AI_STUDYMATE_REFUSAL"
MIN_ANSWER_CONTENT_CHARS = 40
CONTENT_NUDGE_SUFFIX = (
    "\n\n---\n\nYour previous answer was rejected: it contained only citation "
    "tags or repeated a source's title/page/section instead of explaining "
    "anything. Write 2-5 full sentences that actually explain the mechanism "
    "being asked about, in your own words, using only the sources above, "
    "with bracket citations like [S1] woven into the sentences."
)


def match_chitchat_category(text: str) -> str | None:
    normalized = text.strip()
    if CHITCHAT_PATTERNS[0].match(normalized) or CHITCHAT_PATTERNS[1].match(normalized):
        return "greeting"
    if CHITCHAT_PATTERNS[2].match(normalized):
        return "how_are_you"
    if CHITCHAT_PATTERNS[3].match(normalized):
        return "thanks"
    if CHITCHAT_PATTERNS[4].match(normalized):
        return "farewell"
    if CHITCHAT_PATTERNS[5].match(normalized):
        return "who_are_you"
    if CHITCHAT_PATTERNS[6].match(normalized):
        return "what_can_you_help_with"
    return None


def is_chitchat(text: str) -> bool:
    return match_chitchat_category(text) is not None


def chitchat_response(text: str) -> str:
    category = match_chitchat_category(text)
    return CHITCHAT_RESPONSES.get(category, CHITCHAT_RESPONSES["greeting"])


def get_ollama_client(settings: Settings) -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def classify_intent(settings: Settings, user_input: str, client: ollama.Client | None = None) -> str:
    if is_chitchat(user_input):
        return "chitchat"
    try:
        active_client = client or get_ollama_client(settings)
        response = active_client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            options={"temperature": 0, "num_ctx": 8192},
        )
        label = response["message"]["content"].strip().upper()
    except Exception:
        label = "IN_SCOPE"
    if "OUT_OF_SCOPE" in label:
        return "out_of_scope"
    return "in_scope"


def build_source_map(retrieved_chunks: list[dict]) -> dict:
    source_map = {}
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_id = f"S{i}"
        meta = chunk["metadata"]
        source_map[source_id] = {
            "label": f"{meta['title']}, page {meta['page']}, Section: {meta['section']}",
            "chunk": chunk,
        }
    return source_map


def build_prompt(question: str, source_map: dict) -> str:
    context_blocks = []
    for source_id, entry in source_map.items():
        context_blocks.append(f"[{source_id}] {entry['label']}\n{entry['chunk']['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def contains_refusal_sentinel(answer: str) -> bool:
    """True if the model's response contains the refusal token ANYWHERE, not
    just as the entire response. A small local model sometimes writes several
    sentences of half-reasoning and only then gives up and appends
    AI_STUDYMATE_REFUSAL -- an exact `.strip() == REFUSAL_SENTINEL` check
    misses that and lets the half-reasoning (plus the literal sentinel text)
    leak straight into the answer shown to the user."""
    return REFUSAL_SENTINEL in answer


def strip_answer_scaffolding(answer: str, source_map: dict) -> str:
    """Remove citation tags AND any place the model directly typed out a
    source's title/page/section as prose (a rule-3 violation some smaller
    models make). What's left is whatever actual explanatory content the
    model wrote, if any."""
    stripped = CITATION_ID_PATTERN.sub("", answer)
    for entry in source_map.values():
        stripped = stripped.replace(entry["label"], "")
    stripped = re.sub(r"[()\[\]]", "", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def is_citation_only_answer(answer: str, source_map: dict, min_chars: int = MIN_ANSWER_CONTENT_CHARS) -> bool:
    """True if, once citation tags and repeated source labels are stripped
    out, there's essentially no explanatory sentence left -- i.e. the model
    handed back scaffolding instead of an answer."""
    return len(strip_answer_scaffolding(answer, source_map)) < min_chars


def extract_cited_ids(answer: str) -> list[int]:
    return sorted(set(int(m) for m in CITATION_ID_PATTERN.findall(answer)))


def validate_and_render_citations(answer: str, source_map: dict) -> tuple[str | None, str]:
    valid_ids = set(int(source_id[1:]) for source_id in source_map.keys())
    cited_ids = extract_cited_ids(answer)
    if not cited_ids:
        return None, "no_citation_found"
    invalid_ids = [cid for cid in cited_ids if cid not in valid_ids]
    if invalid_ids:
        return None, f"invalid_citation_ids: {invalid_ids}"
    rendered = answer
    for source_id, entry in source_map.items():
        rendered = rendered.replace(f"[{source_id}]", f"({entry['label']})")
    return rendered, "valid"


def call_ollama(settings: Settings, prompt: str, client: ollama.Client | None = None) -> str:
    try:
        active_client = client or get_ollama_client(settings)
        response = active_client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": settings.llm_temperature, "num_ctx": 8192},
        )
        return response["message"]["content"]
    except Exception as e:
        return f"OLLAMA_CALL_FAILED: {e}"


def answer_question(retrieval_context, settings: Settings, question: str, client: ollama.Client | None = None) -> dict:
    from app.services.retrieval import hybrid_retrieve

    intent = classify_intent(settings, question, client=client)

    if intent == "chitchat":
        return {
            "question": question,
            "intent": intent,
            "answer": chitchat_response(question),
            "is_refusal": False,
            "citation_status": "n/a (chitchat)",
            "retrieved_chunks": [],
        }

    if intent == "out_of_scope":
        return {
            "question": question,
            "intent": intent,
            "answer": OUT_OF_SCOPE_TEXT,
            "is_refusal": True,
            "citation_status": "n/a (out_of_scope)",
            "retrieved_chunks": [],
        }

    retrieved = hybrid_retrieve(retrieval_context, settings, question)
    source_map = build_source_map(retrieved)
    prompt = build_prompt(question, source_map)
    raw_answer = call_ollama(settings, prompt, client=client)

    # One retry with a nudge if the model handed back scaffolding (bare
    # citation tags / repeated source labels) instead of an actual answer --
    # cheap, and gives the small model a second shot before we give up.
    if (
        not raw_answer.startswith("OLLAMA_CALL_FAILED")
        and not contains_refusal_sentinel(raw_answer)
        and is_citation_only_answer(raw_answer, source_map)
    ):
        raw_answer = call_ollama(settings, prompt + CONTENT_NUDGE_SUFFIX, client=client)

    if (
        raw_answer.startswith("OLLAMA_CALL_FAILED")
        or contains_refusal_sentinel(raw_answer)
    ):
        final_answer = REFUSAL_TEXT
        is_refusal = True
        citation_status = "n/a (refusal)"
    else:
        rendered, citation_status = validate_and_render_citations(raw_answer, source_map)
        if rendered is None:
            final_answer = REFUSAL_TEXT
            is_refusal = True
        elif is_citation_only_answer(rendered, source_map):
            final_answer = REFUSAL_TEXT
            is_refusal = True
            citation_status = "citation_only_no_content"
        else:
            final_answer = rendered
            is_refusal = False

    return {
        "question": question,
        "intent": intent,
        "answer": final_answer,
        "is_refusal": is_refusal,
        "citation_status": citation_status,
        "retrieved_chunks": retrieved,
    }
