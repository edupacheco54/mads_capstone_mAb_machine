import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from src.utils.device import get_best_device


class ESMCModel:
    """
    Wrapper for generating protein embeddings using the ESM-C model.

    This class loads a pretrained ESM-C model and exposes a simple interface
    for converting a protein sequence into a fixed-length embedding vector.
    The embedding is derived from per-residue representations produced by the
    model and aggregated via mean pooling.

    Parameters
    ----------
    model_name : str, optional
        Name of the pretrained ESM-C checkpoint to load. Common options include
        "esmc_300m" and "esmc_600m".
    device : str, optional
        PyTorch device identifier ("cuda", "mps", or "cpu"). If not provided,
        the best available device is automatically selected using
        `get_best_device()`.

    Notes
    -----
    ESM-C produces per-residue embeddings for each amino acid in the sequence.
    These embeddings are pooled (averaged) to obtain a single vector
    representation for the entire protein. This representation can then be
    used as input features for downstream machine learning models.
    """

    def __init__(self, model_name="esmc_600m", device=None):
        """
        Initialize the ESM-C embedding model.

        Loads the specified pretrained ESM-C model and moves it to the chosen
        computation device.

        Parameters
        ----------
        model_name : str
            Name of the ESM-C checkpoint to load.
        device : str or None
            Device to run the model on. If None, the best available device is
            automatically selected.
        """

        if device is None:
            device = get_best_device()
        self.device = device
        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, sequence):
        """
        Generate a fixed-length embedding for a protein sequence.

        The input sequence is encoded using the ESM-C model to obtain
        per-residue embeddings. These embeddings are then mean-pooled to
        produce a single vector representation for the entire protein.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array representing the pooled protein embedding.
            The dimensionality depends on the specific ESM-C model used.
        """

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self.model.encode(protein)
            output = self.model.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )

            # Per-residue embeddings
            embeddings = output.embeddings.squeeze()

            # Mean pooling to obtain a single protein representation
            pooled = embeddings.mean(dim=0)

        return pooled.cpu().numpy()
