import re
import torch
from transformers import T5EncoderModel, T5Tokenizer

from src.utils.device import get_best_device


class ProstT5Model:
    """
    Wrapper for generating protein embeddings using the ProstT5 model.

    ProstT5 is a protein language model based on the T5 architecture. It can
    encode amino acid sequences into dense vector representations that capture
    structural and functional properties of proteins.

    This wrapper provides a simple interface to convert a protein sequence into
    a fixed-length embedding vector. The model generates per-residue embeddings,
    which are aggregated using mean pooling to produce a single representation
    for the entire protein sequence.

    Parameters
    ----------
    model_name : str, optional
        Name of the pretrained ProstT5 model checkpoint to load. Default is
        "Rostlab/ProstT5".
    device : str, optional
        PyTorch device identifier ("cuda", "mps", or "cpu"). If not provided,
        the best available device is selected automatically using
        `get_best_device()`.

    Notes
    -----
    ProstT5 expects amino acid sequences to be space-separated and prefixed
    with a task token. When embedding amino acid sequences, the "<AA2fold>"
    prefix is used. Rare or ambiguous amino acids (U, Z, O, B) are replaced
    with "X" before tokenization.

    The model produces per-residue embeddings which are then mean-pooled to
    produce a fixed-length vector representation suitable for downstream
    machine learning tasks.
    """

    def __init__(self, model_name="Rostlab/ProstT5", device=None):
        """
        Initialize the ProstT5 embedding model.

        Loads the specified pretrained ProstT5 checkpoint and moves it to the
        selected computation device.

        Parameters
        ----------
        model_name : str
            Name of the ProstT5 model checkpoint to load.
        device : str or None
            Device to run the model on. If None, the best available device
            is automatically selected.
        """

        if device is None:
            device = get_best_device()

        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Use half precision on CUDA GPUs for improved performance
        if self.device == "cuda":
            self.model.half()
        else:
            self.model.float()

    def _prepare_sequence(self, sequence):
        """
        Prepare an amino acid sequence for ProstT5 tokenization.

        This includes:
        - Replacing ambiguous residues (U, Z, O, B) with "X"
        - Adding whitespace between residues
        - Adding the "<AA2fold>" prefix required for ProstT5

        Parameters
        ----------
        sequence : str
            Raw amino acid sequence.

        Returns
        -------
        str
            Formatted sequence ready for tokenization.
        """

        sequence = re.sub(r"[UZOB]", "X", sequence)
        spaced = " ".join(list(sequence))
        return f"<AA2fold> {spaced}"

    def embed(self, sequence):
        """
        Generate a fixed-length embedding for a protein sequence.

        The sequence is first formatted for ProstT5 tokenization and then
        passed through the model to obtain per-residue embeddings. Special
        tokens (such as the prefix token) are excluded, and the remaining
        residue embeddings are averaged to produce a single vector
        representation of the protein.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array containing the pooled protein embedding.
            The embedding dimension depends on the ProstT5 model.
        """

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

        # Skip the prefix token and select only residue embeddings
        residue_embeddings = embedding_repr.last_hidden_state[0, 1 : seq_len + 1]

        # Mean pooling to obtain a single protein representation
        pooled = residue_embeddings.mean(dim=0)

        return pooled.float().cpu().numpy()
