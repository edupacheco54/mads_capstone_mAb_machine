import re
import torch
from transformers import T5EncoderModel, T5Tokenizer

from src.utils.device import get_best_device


class ProtT5Model:
    """
    Wrapper for generating protein embeddings using the ProtT5 model.

    ProtT5 is a protein language model based on the T5 architecture trained on
    large protein sequence datasets. It produces contextual embeddings for each
    amino acid residue in a sequence.

    This wrapper converts a protein sequence into a fixed-length embedding by
    generating per-residue representations and applying mean pooling across
    residues to produce a single vector representation for the entire protein.

    Parameters
    ----------
    model_name : str, optional
        Name of the pretrained ProtT5 model checkpoint. Default is
        "Rostlab/prot_t5_xl_uniref50".
    device : str, optional
        PyTorch device identifier ("cuda", "mps", or "cpu"). If not provided,
        the best available device is automatically selected using
        `get_best_device()`.

    Notes
    -----
    ProtT5 expects sequences to be formatted as space-separated amino acids.
    Ambiguous or rare residues (U, Z, O, B) are replaced with "X" before
    tokenization.

    The model produces per-residue embeddings which are averaged to obtain
    a single fixed-length protein representation suitable for downstream
    machine learning models.
    """

    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device=None):
        """
        Initialize the ProtT5 embedding model.

        Loads the specified pretrained ProtT5 checkpoint and moves it to the
        selected computation device.

        Parameters
        ----------
        model_name : str
            Name of the ProtT5 model checkpoint to load.
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

    def _prepare_sequence(self, sequence):
        """
        Prepare an amino acid sequence for ProtT5 tokenization.

        This includes:
        - Replacing ambiguous residues (U, Z, O, B) with "X"
        - Adding whitespace between residues

        Parameters
        ----------
        sequence : str
            Raw amino acid sequence.

        Returns
        -------
        str
            Formatted sequence ready for ProtT5 tokenization.
        """

        sequence = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(list(sequence))

    def embed(self, sequence):
        """
        Generate a fixed-length embedding for a protein sequence.

        The sequence is tokenized and passed through the ProtT5 encoder to
        obtain contextual embeddings for each residue. These per-residue
        embeddings are then mean-pooled to produce a single vector
        representation for the protein.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array containing the pooled protein embedding.
            The dimensionality depends on the ProtT5 model configuration.
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

        # Extract embeddings corresponding to the original residues
        residue_embeddings = embedding_repr.last_hidden_state[0, :seq_len]

        # Mean pooling to produce a single protein representation
        pooled = residue_embeddings.mean(dim=0)

        return pooled.cpu().numpy()
