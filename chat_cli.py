"""대화 생성/채팅 함수 (노트북과 CLI 공용)."""
import torch

def reply(model, tok, history, user_msg, max_new_tokens=160, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.15):
    msgs = history + [{"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(msgs[-6:], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([[tok.bos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"]])
    end_id = tok.convert_tokens_to_ids("<|end|>")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                             eos_token_id=[end_id, tok.eos_token_id], pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return text
