from transformers import PretrainedConfig


class NeuroBrainConfig(PretrainedConfig):
    model_type = "neurobrain"
    auto_map = {"AutoConfig": "configuration_neurobrain.NeuroBrainConfig", "AutoModel": "modeling_neurobrain.NeuroBrainModel", "AutoModelForCausalLM": "modeling_neurobrain.NeuroBrainForCausalLM"}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=8192, hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                 num_key_value_heads=None, intermediate_ratio=4, max_position_embeddings=2048,
                 img_size=32, patch=8, n_mels=32, audio_patch=4, video_frames=4,
                 mem_slots=64, gate_keep=0.75, dropout=0.0, rms_eps=1e-6,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=True,
                 grad_ckpt=False, mlp_type='dense', k_frac=0.25, fast_weights=False, fw_heads=4, homeostasis=0.0, visual_hierarchy=False, **kw):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or max(1, num_attention_heads // 2)
        self.intermediate_ratio = intermediate_ratio
        self.max_position_embeddings = max_position_embeddings
        self.img_size, self.patch, self.n_mels = img_size, patch, n_mels
        self.audio_patch, self.video_frames = audio_patch, video_frames
        self.mem_slots, self.gate_keep, self.dropout, self.rms_eps = mem_slots, gate_keep, dropout, rms_eps
        self.grad_ckpt = grad_ckpt
        self.visual_hierarchy = visual_hierarchy
        self.mlp_type, self.k_frac, self.fast_weights, self.fw_heads, self.homeostasis = mlp_type, k_frac, fast_weights, fw_heads, homeostasis
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kw)
