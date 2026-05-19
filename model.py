import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 国内魔搭，不连 huggingface，绝对不超时！
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate(prompt, num_return_sequences=1, temperature=0.7, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for output in outputs:
        resp = tokenizer.decode(output, skip_special_tokens=True)
        resp = resp[len(input_text):].strip()
        results.append(resp)
    return results