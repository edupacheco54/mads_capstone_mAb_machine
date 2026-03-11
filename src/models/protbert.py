import re
import torch
from transformers import BertModel, BertTokenizer

from src.utils.device import get_best_device


class ProtBERTModel:
    def __init__(self, model_name="Rostlab/prot_bert", device=None):
        if device is None:
            device = get_best_device()

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _prepare_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(list(sequence))

    def embed(self, sequence):
        formatted_sequence = self._prepare_sequence(sequence)

        encoded_input = self.tokenizer(
            formatted_sequence, return_tensors="pt", add_special_tokens=True
        )

        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        seq_len = len(sequence)

        residue_embeddings = output.last_hidden_state[0, 1 : seq_len + 1]

        pooled = residue_embeddings.mean(dim=0)

        return pooled.cpu().numpy()
