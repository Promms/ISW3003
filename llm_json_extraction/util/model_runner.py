import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


class ModelRunner:
    def __init__(self, model_id=MODEL_ID, device=None, enable_thinking=False):
        self.device = device or pick_device()
        self.enable_thinking = enable_thinking
        print(f"[model] loading '{model_id}' (device={self.device}) ...")
        print("[model] the first run downloads the model (~3-4 GB); "
              "Hugging Face shows download progress below.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto" if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[model] ready.")

    @torch.no_grad()
    def chat(self, messages, max_new_tokens=256):
        """messages = [{"role": "user"/"assistant", "content": ...}, ...]

        Decodes and returns only the newly generated tokens as a string.
        Uses greedy decoding (do_sample=False) for reproducibility.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,  # Qwen3-specific
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Keep only the tokens generated after the prompt.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
