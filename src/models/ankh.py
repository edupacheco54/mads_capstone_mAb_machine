import torch
import ankh

from src.utils.device import get_best_device


class AnkhModel:
    def __init__(self, model_size="base", device=None):
        if device is None:
            device = get_best_device()

        self.device = device

        if model_size == "base":
            self.model, self.tokenizer = ankh.load_base_model()
        elif model_size == "large":
            self.model, self.tokenizer = ankh.load_large_model()
        else:
            raise ValueError("model_size must be 'base' or 'large'")

        self.model = self.model.to(self.device)
        self.model.eval()

    def embed(self, sequence):
        outputs = self.tokenizer(
            [list(sequence)],
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        )

        input_ids = outputs["input_ids"].to(self.device)
        attention_mask = outputs["attention_mask"].to(self.device)

        with torch.no_grad():
            embedding_repr = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        embeddings = embedding_repr.last_hidden_state[0]

        mask = attention_mask[0].bool()
        residue_embeddings = embeddings[mask]

        pooled = residue_embeddings.mean(dim=0)

        return pooled.cpu().numpy()
