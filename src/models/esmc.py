import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from src.utils.device import get_best_device


class ESMCModel:
    def __init__(self, model_name="esmc_600m", device=None):
        if device is None:
            device = get_best_device()
        self.device = device
        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, sequence):
        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self.model.encode(protein)
            output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            embeddings = output.embeddings.squeeze()
            pooled = embeddings.mean(dim=0)

        return pooled.cpu().numpy()