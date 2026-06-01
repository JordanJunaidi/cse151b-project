import os, re, csv, json
from pathlib import Path
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_DISABLE_DEEP_GEMM"] = "1"


MODEL_ID       = "Qwen/Qwen3-4B-Thinking-2507"
PRIV_PATH      = "data/private.jsonl"
MAX_LORA_RANK  = 16
N_VOTE         = 3
GEN_MAX_TOKENS = 8192

BEST_ADAPTER   = "TheAsianFish/qwen3-math-lora"

SUB_JSONL      = "results/private_final.jsonl"
OUTPUT_CSV     = "submission.csv"

# ---- Prompts (verbatim from notebook Phase 3 / Cell I) ----
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Solve the problem step-by-step, but be concise, no repetition, no over-explanation. "
    "Think about the problem type first, then apply the correct method. "
    "Give the final answer in exact form: keep symbolic constants such as pi, e, and square "
    "roots, and exact fractions. Do not convert to decimals. "
    "If the answer is a family of solutions, include the free integer parameter exactly as the "
    "problem names it (for example k). Use the exact notation and variable names the problem uses. "
    "If the problem shows [ANS], your boxed answer is whatever replaces [ANS]. "
    "Before finalizing, verify your answer is correct. "
    "Your response MUST end with \\boxed{answer} as the absolute last thing you write. "
    "For multiple sub-answers in order, use \\boxed{a, b, c}. "
    "Never write anything after the boxed answer."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and all options carefully. Eliminate wrong answers, then select the best one. "
    "You MUST always select one of the given options, never say none are correct. "
    "Your response MUST end with \\boxed{X} where X is the letter only. "
    "Never write anything after the boxed answer."
)

def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts}"
    return SYSTEM_PROMPT_MATH, question

def extract_boxed(text):
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return ""
    i, depth, out = starts[-1], 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: break
        out.append(c); i += 1
    return "".join(out).strip()

def extract_mcq_letter(text):
    b = extract_boxed(text)
    m = re.match(r"\s*([A-Za-z])\s*$", b)
    return m.group(1).upper() if m else ""

def vote(samples, item):
    keyf = extract_mcq_letter if item.get("options") else extract_boxed
    answers = [keyf(s) for s in samples]
    valid = [a for a in answers if a]
    if not valid:
        return samples[0]
    maj = Counter(valid).most_common(1)[0][0]
    for s, a in zip(samples, answers):
        if a == maj:
            return s
    return samples[0]


def run_inference():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("GPU:", torch.cuda.get_device_name(0))

    # ---- Load model with LoRA enabled (notebook Phase 3 / Cell I) ----
    llm = LLM(model=MODEL_ID, enable_lora=True, max_lora_rank=MAX_LORA_RANK,
              dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=8192,
              enable_prefix_caching=True, trust_remote_code=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()

    def prompt_str_for(item):
        system, user = build_prompt(item["question"], item.get("options"))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    print("Using adapter:", BEST_ADAPTER)
    priv = [json.loads(l) for l in open(PRIV_PATH)]
    Path(SUB_JSONL).parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if Path(SUB_JSONL).exists():
        for l in open(SUB_JSONL):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    todo = [d for d in priv if d["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go (of {len(priv)})")

    sp  = SamplingParams(n=N_VOTE, temperature=0.7, top_p=0.95, top_k=20,
                         max_tokens=GEN_MAX_TOKENS, repetition_penalty=1.0)
    req = None if BEST_ADAPTER is None else LoRARequest("best", 1, BEST_ADAPTER)
    CHUNK = 64
    with open(SUB_JSONL, "a") as f:
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s:s+CHUNK]
            outs = llm.generate([prompt_str_for(it) for it in chunk], sp, lora_request=req)
            for it, o in zip(chunk, outs):
                chosen = vote([c.text.strip() for c in o.outputs], it)
                f.write(json.dumps({"id": it["id"], "response": chosen}) + "\n")
            f.flush()
            print(f"  {min(s+CHUNK, len(todo))}/{len(todo)}")
    print("Private inference complete.")

    # ---- Write submission.csv (notebook Phase 3 / Cell L) ----
    order = {d["id"]: k for k, d in enumerate(priv)}
    rows = [json.loads(l) for l in open(SUB_JSONL)]
    rows.sort(key=lambda r: order.get(r["id"], 10**9))

    assert len(rows) == len(priv), f"expected {len(priv)} rows, got {len(rows)}"
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "response"])
        for r in rows:
            w.writerow([r["id"], r["response"]])
    print(f"Wrote {OUTPUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    run_inference()