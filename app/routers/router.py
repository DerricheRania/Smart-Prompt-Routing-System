"""
Routing logic for the Smart Prompt Router.

Flow: prompt -> predict_intent() -> matching handler -> response

Models are loaded lazily (on first use, not on import) so that:
  - the FastAPI server starts instantly instead of blocking on model downloads
  - if one model fails to download/load, the others still work
  - the very first request for a given intent will be slower (cold start),
    every request after that is fast since the model stays cached in memory

NOTE on translation / summarization:
Newer `transformers` releases (v5+) removed the "translation", "summarization",
and "text2text-generation" task strings from the pipeline() registry, so
pipeline("translation", ...) / pipeline("summarization", ...) raise a KeyError
on those versions. To stay correct across transformers versions, those 2 seq2seq
models are loaded directly with AutoTokenizer + AutoModelForSeq2SeqLM and called
with .generate() instead of pipeline(). Sentiment still uses pipeline() since its
task name ("text-classification") remains registered either way.

NOTE on chat / qa:
Both now share microsoft/Phi-4-mini-instruct (3.8B params, MIT license) instead
of DialoGPT-small (chat) and flan-t5-base (qa). DialoGPT and flan-t5-base are both
too small/dated to reliably follow instructions or recall facts — Phi-4-mini-instruct
is a modern instruction-tuned model that's still practical to run on CPU (no GPU
required), with much better factual recall and conversational coherence. The two
intents use the SAME loaded model (cached once, under the "chat_qa" key) but with
different prompting: qa sends a single question, chat sends the running per-session
message history through the model's real chat template.

Heads-up: Phi-4-mini-instruct needs around 8GB of RAM/disk for its bf16 weights
and is slower per request on CPU than the small models used before — see README
for details.
"""

import logging
import re

import joblib
import torch

from app.config.config import WEIGHTS_PATH

logger = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Intent classifier (loaded eagerly — it's a tiny sklearn pipeline, instant)
# ---------------------------------------------------------------------------
classifier = joblib.load(f"{WEIGHTS_PATH}/intent_classifier.pkl")
label_encoder = joblib.load(f"{WEIGHTS_PATH}/label_encoder.pkl")

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache for lazily-loaded models: {name: pipeline-or-(tokenizer, model)}
_model_cache: dict = {}

# Per-session chat history for multi-turn conversation, keyed by session_id.
# Stores a list of {"role": ..., "content": ...} messages (Phi-4 chat template format).
_chat_history: dict = {}


def predict_intent(text: str) -> str:
    """Predict intent label: translation / summarization / sentiment / qa / chat."""
    pred = classifier.predict([text])[0]
    return label_encoder.inverse_transform([pred])[0]


def _get_model(name: str):
    """Lazily build and cache a model/pipeline by name."""
    if name in _model_cache:
        return _model_cache[name]

    logger.info(f"Loading model '{name}' for the first time (this may take a while)...")

    if name == "translation":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(TORCH_DEVICE)
        model.eval()
        result = (tokenizer, model)

    elif name == "summarization":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # t5-small (60M params) is too weak for real summarization — it tends to
        # either echo short input back verbatim or degenerate into near-gibberish
        # on longer input. t5-base (220M) is meaningfully better at actually
        # compressing text while still being practical to run on CPU.
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base").to(TORCH_DEVICE)
        model.eval()
        result = (tokenizer, model)

    elif name == "sentiment":
        from transformers import pipeline

        result = pipeline(
            "text-classification",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=DEVICE,
        )

    elif name == "chat_qa":
        # Shared model for both "chat" and "qa" intents. microsoft/Phi-4-mini-instruct
        # is a modern 3.8B instruction-tuned model (MIT license) — a big step up from
        # DialoGPT-small (chat, no instruction-following, 2019-era) and flan-t5-base
        # (qa, too small to reliably recall facts). It needs more RAM/disk (~8GB at
        # bf16) and is slower per request on CPU, but answers are far more reliable.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            torch_dtype=torch.bfloat16 if TORCH_DEVICE == "cpu" else "auto",
            trust_remote_code=True,
        ).to(TORCH_DEVICE)
        model.eval()
        result = (tokenizer, model)

    else:
        raise ValueError(f"Unknown model name: {name}")

    _model_cache[name] = result
    return result


def preload_models() -> None:
    """
    Optionally call this at startup to download/load all models up front
    instead of paying the cold-start cost on each intent's first request.
    Not called automatically — see main.py for how to enable it.
    """
    for name in ("translation", "summarization", "sentiment", "chat_qa"):
        _get_model(name)


# ---------------------------------------------------------------------------
# Translation prompt cleanup
# ---------------------------------------------------------------------------

# OpusMT is a pure translation model with no instruction-following: if you feed it
# 'Translate "Hello" into French' it will translate the WHOLE sentence literally,
# instructions and quote marks included. These patterns strip the instruction
# wrapper so only the text that actually needs translating reaches the model.
_QUOTE_CHARS = '"\u201c\u201d\u2018\u2019\''

_TRANSLATION_WRAPPERS = [
    # Translate "X" into/to/in Y   OR   Translate 'X' into/to/in Y
    r'^(?:please\s+)?translate\s+(["\u201c\u2018\'])(.+)\1\s+(?:into|to|in)\s+\w+\.?$',
    # How do you say "X" / 'X' in Y
    r'^how\s+do\s+you\s+say\s+(["\u201c\u2018\'])(.+)\1\s+in\s+\w+\??$',
    # How do you say X in Y (no quotes)
    r'^how\s+do\s+you\s+say\s+(.+?)\s+in\s+\w+\??$',
    # Translate this sentence/text/paragraph (from X) to/into/in Y: <text>
    r'^(?:please\s+)?translate\s+this\s+(?:sentence|text|paragraph)\s*(?:from\s+\w+\s+)?(?:to|into|in)\s+\w+\s*[:\-]\s*(.+)$',
    # Translate: <text>
    r'^(?:please\s+)?translate\s*[:\-]\s*(.+)$',
    # Translate to/into Y: <text>  OR  Translate to/into Y <text>
    r'^(?:please\s+)?translate\s+(?:to|into|in)\s+\w+\s*[:\-]?\s*(.+)$',
    # Translate <text> to/into/in Y  (text then language, no quotes, no colon)
    r'^(?:please\s+)?translate\s+(.+?)\s+(?:into|to|in)\s+\w+\.?$',
]


def _extract_text_to_translate(prompt: str) -> str:
    """Strip an instruction wrapper like 'Translate "X" into French' down to just X."""
    stripped = prompt.strip()
    for pattern in _TRANSLATION_WRAPPERS:
        match = re.match(pattern, stripped, flags=re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            extracted = groups[-1].strip().strip(_QUOTE_CHARS).strip()
            if extracted and extracted != ".":
                return extracted
    # No wrapper matched (or nothing meaningful extracted) — translate as-is.
    return stripped


# ---------------------------------------------------------------------------
# Per-intent handlers
# ---------------------------------------------------------------------------

def gen_translation(text: str) -> str:
    tokenizer, model = _get_model("translation")
    text_to_translate = _extract_text_to_translate(text)
    inputs = tokenizer(text_to_translate, return_tensors="pt", truncation=True).to(TORCH_DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def gen_summary(text: str) -> str:
    """Summarize with parameters tuned to actually compress rather than echo."""
    if len(text.split()) < 15:
        return text  # too short to meaningfully summarize

    tokenizer, model = _get_model("summarization")
    inputs = tokenizer(
        "summarize: " + text, return_tensors="pt", truncation=True, max_length=512
    ).to(TORCH_DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=80,
            min_length=10,
            do_sample=False,
            no_repeat_ngram_size=3,
            num_beams=4,
            length_penalty=2.0,    # rewards shorter, more compressed output
            repetition_penalty=1.3,  # guards against degenerate loops on long input
            early_stopping=True,
        )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Sanity checks: if generation degenerated (too short / mostly repeated /
    # not actually shorter than the input), fall back to a sentence-aware
    # truncation so the user sees something useful instead of garbage.
    looks_degenerate = (
        len(result.split()) < 3
        or len(result.split()) >= len(text.split())
        or len(set(result.lower().split())) <= 2  # e.g. "you ! you !"
    )
    if looks_degenerate:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        keep = max(1, len(sentences) // 2)
        result = " ".join(sentences[:keep]).strip()
        if not result:
            words = text.split()
            result = " ".join(words[: max(8, len(words) // 2)]) + "..."
    return result


def gen_sentiment(text: str) -> str:
    pipe = _get_model("sentiment")
    result = pipe(text)[0]
    return f"{result['label']} ({result['score']:.2f})"


def gen_qa(text: str) -> str:
    """Answer a factual question with no external context required, using Phi-4-mini-instruct."""
    tokenizer, model = _get_model("chat_qa")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer factual questions accurately "
                       "and concisely, in one or two sentences.",
        },
        {"role": "user", "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(TORCH_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    if not answer:
        return "I'm not confident enough to answer that one."
    return answer


def gen_chat(text: str, session_id: str = "default") -> str:
    """
    Multi-turn chat using Phi-4-mini-instruct's real chat template. Conversation
    history is kept per session_id as a list of {role, content} messages, so
    follow-up messages in the same chat have proper context.
    """
    tokenizer, model = _get_model("chat_qa")

    history = _chat_history.get(session_id)
    if history is None:
        history = [
            {"role": "system", "content": "You are a friendly, helpful assistant. Keep replies concise."}
        ]

    history.append({"role": "user", "content": text})

    inputs = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(TORCH_DEVICE)

    # Keep the conversation from growing unbounded (measured in input tokens)
    max_history_tokens = 1024
    if inputs["input_ids"].shape[-1] > max_history_tokens:
        # Drop oldest turns (but keep the system message at index 0)
        while len(history) > 2 and inputs["input_ids"].shape[-1] > max_history_tokens:
            history.pop(1)
            inputs = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(TORCH_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.8,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    if not reply:
        reply = "Hmm, I'm not sure how to respond to that — could you rephrase?"

    history.append({"role": "assistant", "content": reply})
    _chat_history[session_id] = history
    return reply


def reset_chat_session(session_id: str = "default") -> None:
    """Clear stored conversation history for a session (e.g. when the user hits 'Back')."""
    _chat_history.pop(session_id, None)


router = {
    "translation": gen_translation,
    "summarization": gen_summary,
    "sentiment": gen_sentiment,
    "qa": gen_qa,
    "chat": gen_chat,
}


def route_prompt(text: str, session_id: str = "default"):
    """
    Full agent: detect intent, call the matching model, and return (intent, response).
    If the specialized model fails for any reason (e.g. download issue), falls back
    to the chat handler so the user still gets a reply instead of a 500 error.
    """
    intent = predict_intent(text)
    handler = router[intent]

    try:
        if intent == "chat":
            response = handler(text, session_id=session_id)
        else:
            response = handler(text)
    except Exception as exc:
        logger.exception(f"Handler for intent '{intent}' failed, falling back to chat")
        try:
            response = gen_chat(text, session_id=session_id)
            intent = "chat"
        except Exception:
            response = (
                "Sorry, I couldn't process that prompt right now "
                f"(error in '{intent}' handler: {exc})."
            )

    return intent, response


# ---------- Optional CLI for quick demo ----------
if __name__ == "__main__":
    print("Smart Prompt Router (CLI)")
    print("Type 'quit' to exit.\n")
    while True:
        user = input("Prompt: ")
        if user.strip().lower() in {"quit", "exit"}:
            break
        intent, answer = route_prompt(user)
        print(f"[{intent}] {answer}\n")