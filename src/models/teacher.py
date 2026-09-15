import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMTeacher:
    EMOTION_LABELS_27 = [
        "admiration","amusement","anger","annoyance","approval","caring","confusion",
        "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
        "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
        "pride","realization","relief","remorse","sadness","surprise"
    ]

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):
        print(f"Loading teacher model: {model_name} on {device}")
        self.device = torch.device(device)
        # Load directly on CPU with NO offloading (avoids 'meta device' bug)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute first-token IDs for each emotion label
        self.emotion_token_ids = []
        for lbl in self.EMOTION_LABELS_27:
            ids = self.tokenizer.encode(" " + lbl, add_special_tokens=False)
            self.emotion_token_ids.append(ids[0] if ids else 0)

    @torch.no_grad()
    def generate_soft_labels(self, texts, temperature=2.0, batch_size=4):
        all_probs = []
        all_hidden = []
        template = (
            "You are an emotion classifier. Read the text and output ONLY "
            "the single best matching emotion word from this list:\n"
            "{labels}\n\nText: {text}\nEmotion:"
        )
        labels_str = ", ".join(self.EMOTION_LABELS_27)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            prompts = [template.format(labels=labels_str, text=t) for t in batch]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(self.device)

            out = self.model(**inputs, output_hidden_states=True)

            # Mean-pooled hidden state from last layer
            last = out.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1)
            all_hidden.append(pooled.cpu())

            # Emotion logits at the final position
            final_logits = out.logits[:, -1, :]  # [B, vocab]
            emo_logits = final_logits[:, self.emotion_token_ids]  # [B, 27]
            probs = F.softmax(emo_logits / temperature, dim=-1)
            all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0), torch.cat(all_hidden, dim=0)
