"""Build a BPE tokenizer (PreTrainedTokenizerFast) with a chat template, then save."""
import sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>\\n' }}"
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}"
    "{% if p['type'] == 'text' %}{{ p['text'] }}"
    "{% elif p['type'] == 'image' %}{{ '<|image|>' }}"
    "{% elif p['type'] == 'audio' %}{{ '<|audio|>' }}"
    "{% elif p['type'] == 'video' %}{{ '<|video|>' }}"
    "{% endif %}{% endfor %}{% endif %}"
    "{{ '<|end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|image|>", "<|audio|>", "<|video|>"]


def build(corpus_iter, vocab_size=4096, out="tokenizer_out"):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus_iter, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
                                   additional_special_tokens=SPECIAL[4:], chat_template=CHAT_TEMPLATE, clean_up_tokenization_spaces=False)
    fast.save_pretrained(out)
    return fast
