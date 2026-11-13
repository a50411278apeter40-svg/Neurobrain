"""사람 점수 기반 강화학습(RLHF) - TRL RLOO.
모델이 답을 만들면 사람이 직접 점수(1~5)를 입력하고, 그 점수가 보상이 되어 모델이 업데이트된다.
(최신 TRL에는 PPOTrainer가 없어, PPO의 후속인 RLOO를 사용한다. 보상 함수에 사람 입력을 직접 연결한다.)"""
import json, random, sys, os, urllib.request

def ask_score(prompt, completion, lo=1, hi=5, input_fn=input):
    """터미널/노트북에서 사람에게 점수를 받는다. 잘못된 입력이면 다시 묻고, 's'는 건너뛰기(중립 보상)."""
    print("=" * 60); print("[질문]", prompt[-300:]); print("[모델 답]", completion[:400] or "(빈 답변)")
    while True:
        v = input_fn(f"점수 입력 ({lo}=아주 나쁨 ~ {hi}=아주 좋음, s=건너뛰기, q=학습 중단): ").strip().lower()
        if v == "q": raise KeyboardInterrupt
        if v == "s": return None
        try:
            x = float(v)
            if lo <= x <= hi: return x
        except ValueError: pass
        print(f"  {lo}~{hi} 사이 숫자를 입력해 주세요.")

def make_human_reward(lo=1, hi=5, log_path="human_scores.jsonl", input_fn=input):
    """TRL reward_funcs용 함수. 점수를 [-1, +1]로 정규화한다(3점=0). 모든 평가는 파일에 기록한다."""
    mid = (lo + hi) / 2; half = (hi - lo) / 2
    def human_reward(prompts, completions, **kw):
        out = []
        for p, c in zip(prompts, completions):
            ptxt = p if isinstance(p, str) else " ".join(m.get("content", "") for m in p)
            ctxt = c if isinstance(c, str) else " ".join(m.get("content", "") for m in c)
            s = ask_score(ptxt, ctxt, lo, hi, input_fn)
            r = 0.0 if s is None else (s - mid) / half
            with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps({"prompt": ptxt, "completion": ctxt, "score": s, "reward": r}, ensure_ascii=False) + "\n")
            out.append(r)
        return out
    return human_reward

def fable_prompts(repo="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", path="5k/train.jsonl", n=64, max_chars=400, max_bytes=40_000_000, token=None):
    """Fable 계열 데이터에서 첫 user 지시문 앞부분을 프롬프트로 뽑는다(스트리밍으로 앞부분만 읽음)."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    h = {"Range": f"bytes=0-{max_bytes}"}
    if token: h["Authorization"] = f"Bearer {token}"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read().decode("utf-8", "ignore").split("\n")[:-1]
    out = []
    for line in raw:
        try: r = json.loads(line)
        except Exception: continue
        for m in r.get("messages", []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                out.append(m["content"][:max_chars]); break
        if len(out) >= n: break
    return out

def chat_prompts(path="chat_all.json", n=64, max_len=120, seed=0):
    rows = json.load(open(path, encoding="utf-8")); random.Random(seed).shuffle(rows)
    return [u for _, u, a in rows if len(u) < max_len][:n]
