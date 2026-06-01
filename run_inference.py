import os
import json
import re
import pandas as pd
from typing import Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 1. Config
BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_HUB_PATH = "JordanJunaidi/qwen3-math-lora"
DATA_PATH = "data/private.jsonl"
OUTPUT_PATH = "submission.csv"

# 2. Prompts
SYSTEM_PROMPT_MCQ = (
    "You are an expert at solving multiple-choice math problems. "
    "Read the problem and all answer choices carefully, reason step by step, "
    "then determine which single option is correct.\n\n"
    "OUTPUT FORMAT RULES (strict):\n"
    "1. After your reasoning, end with a line of the exact form:\n"
    "   Final answer: \\boxed{X}\n"
    "   where X is a single capital letter, with NO spaces, NO punctuation, NO \\text{}.\n"
    "2. Correct:  Final answer: \\boxed{A}    Final answer: \\boxed{F}\n"
    "3. WRONG:    \\boxed{ A }   \\boxed{A.}   \\boxed{\\text{A}}   \\boxed{Option C}\n"
    "4. Do not write anything after the boxed final answer."
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step.\n\n"
    "The question contains one or more [ANS] placeholders marking where answers go.\n\n"
    "OUTPUT FORMAT RULES (strict):\n"
    "1. After your reasoning, end with a line of the form:\n"
    "   Final answer: \\boxed{...} \\boxed{...} ...\n"
    "   with one \\boxed{} per [ANS], in order, separated by single spaces.\n"
    "2. Each \\boxed{} contains exactly ONE answer. Do NOT put multiple answers separated by commas in one box.\n"
    "3. If an answer itself contains commas, keep it inside ONE box: \\boxed{(3, 5)} is one answer.\n"
    "4. Put all \\boxed{} answers together on the final line with nothing between them except spaces.\n"
    "5. Do not write anything after the final \\boxed{}."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        last = labels[-1]
        user = (
            f"Solve this problem. Choose one option (A through {last}).\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opts_text}"
        )
        return SYSTEM_PROMPT_MCQ, user

    n_blanks = max(1, question.count("[ANS]"))
    hint = (
        f"This problem has {n_blanks} [ANS] placeholder{'s' if n_blanks > 1 else ''}, "
        f"so output exactly {n_blanks} \\boxed{{}} block{'s' if n_blanks > 1 else ''} "
        f"together on the final line."
    )
    user = f"{hint}\n\nQuestion:\n{question}"
    return SYSTEM_PROMPT_MATH, user

# 3. Post-Processing / Repair Logic
def _all_boxes_in_order(text):
    out, start = [], 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0: break
        bs = idx + len("\\boxed{")
        depth, i = 1, bs
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            out.append((idx, i, text[bs:i - 1]))
        start = i
    return out

def _extract_all_boxed(text):
    entries = [(a, b, c) for (a, b, c) in _all_boxes_in_order(text) if c]
    if not entries: return []
    last = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r'^[\s,\$\.\;\:\-\&\\]*$', gap):
            last.insert(0, entries[j])
        else: break
    return [e[2] for e in last]

def repair_response(text: str) -> str:
    think_end = text.rfind("</think>")
    region = text[think_end + len("</think>"):] if think_end >= 0 else text
    boxes = [c for (_, _, c) in _all_boxes_in_order(region)]
    if not boxes: 
        boxes = [c for (_, _, c) in _all_boxes_in_order(text)]
    if not boxes: 
        return text
    if _extract_all_boxed(text) == boxes: 
        return text
    canonical = "Final answer: " + " ".join("\\boxed{" + b + "}" for b in boxes)
    return text.rstrip() + "\n\n" + canonical

# 4. Main Entry Point
def run_inference():
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Loading data...")
    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Build prompts
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))

    # Initialize vLLM with LoRA enabled
    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        tokenizer=BASE_MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=.90,
        max_model_len=20480,
        max_num_seqs=64,
        max_num_batched_tokens=32768,
    )

    # Use exact generation hyperparameters from the notebook
    sampling_params = SamplingParams(
        n=1, max_tokens=32768, temperature=0.2, top_p=0.2, top_k=20, min_p=0.0
    )

    # Load adapter directly from HuggingFace Hub
    lora_req = LoRARequest("math_lora", 1, ADAPTER_HUB_PATH)

    print(f"Running inference on {len(prompts)} questions...")
    outs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_req, use_tqdm=True)
    
    raw_responses = [o.outputs[0].text.strip() for o in outs]

    print("Applying post-processing answer repair...")
    repaired_responses = [repair_response(r) for r in raw_responses]

    print(f"Saving final results to {OUTPUT_PATH}...")
    df = pd.DataFrame({
        "id": [d["id"] for d in data],
        "response": repaired_responses
    })
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    run_inference()