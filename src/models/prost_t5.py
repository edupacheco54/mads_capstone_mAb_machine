import re
import torch
from transformers import T5EncoderModel, T5Tokenizer

from src.utils.device import get_best_device


class ProstT5Model:
    def __init__(self, model_name="Rostlab/ProstT5", device=None):
        if device is None:
            device = get_best_device()

        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.device == "cuda":
            self.model.half()
        else:
            self.model.float()

    def _prepare_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", sequence)
        spaced = " ".join(list(sequence))
        return f"<AA2fold> {spaced}"

    def embed(self, sequence):
        formatted_sequence = self._prepare_sequence(sequence)

        ids = self.tokenizer.batch_encode_plus(
            [formatted_sequence],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids["attention_mask"].to(self.device)

        with torch.no_grad():
            embedding_repr = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        seq_len = len(sequence)

        residue_embeddings = embedding_repr.last_hidden_state[0, 1 : seq_len + 1]

        pooled = residue_embeddings.mean(dim=0)

        return pooled.float().cpu().numpy()
