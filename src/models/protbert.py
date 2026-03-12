import re
import torch
from transformers import BertModel, BertTokenizer

from src.utils.device import get_best_device


class ProtBERTModel:
    """
    Wrapper for generating protein embeddings using the ProtBERT model.

    ProtBERT is a protein language model based on the BERT architecture,
    trained on large protein sequence databases. It produces contextual
    embeddings for each amino acid residue in a protein sequence.

    This wrapper converts a protein sequence into a fixed-length embedding
    by generating per-residue representations and applying mean pooling
    across residues to produce a single vector representation for the
    entire protein.

    Parameters
    ----------
    model_name : str, optional
        Name of the pretrained ProtBERT checkpoint. Default is
        "Rostlab/prot_bert".
    device : str, optional
        PyTorch device identifier ("cuda", "mps", or "cpu"). If not provided,
        the best available device is selected automatically using
        `get_best_device()`.

    Notes
    -----
    ProtBERT expects sequences to be formatted as space-separated amino acids.
    Ambiguous or rare residues (U, Z, O, B) are replaced with "X" before
    tokenization.

    The tokenizer adds special tokens such as [CLS] and [SEP]. The [CLS] token
    appears at the beginning of the sequence, so residue embeddings begin at
    position 1 in the model output.
    """

    def __init__(self, model_name="Rostlab/prot_bert", device=None):
        """
        Initialize the ProtBERT embedding model.

        Loads the specified pretrained ProtBERT checkpoint and moves it to the
        selected computation device.

        Parameters
        ----------
        model_name : str
            Name of the ProtBERT model checkpoint to load.
        device : str or None
            Device to run the model on. If None, the best available device
            is automatically selected.
        """

        if device is None:
            device = get_best_device()

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _prepare_sequence(self, sequence):
        """
        Prepare an amino acid sequence for ProtBERT tokenization.

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
            Formatted sequence ready for ProtBERT tokenization.
        """

        sequence = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(list(sequence))

    def embed(self, sequence):
        """
        Generate a fixed-length embedding for a protein sequence.

        The sequence is tokenized and passed through the ProtBERT model to
        obtain contextual embeddings for each residue. The special [CLS]
        token at the beginning of the sequence is excluded, and the remaining
        residue embeddings are mean-pooled to produce a single vector
        representation for the protein.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array containing the pooled protein embedding.
            The embedding dimension depends on the ProtBERT model.
        """

        formatted_sequence = self._prepare_sequence(sequence)

        encoded_input = self.tokenizer(
            formatted_sequence, return_tensors="pt", add_special_tokens=True
        )

        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        seq_len = len(sequence)

        # Skip the [CLS] token at position 0 and select residue embeddings
        residue_embeddings = output.last_hidden_state[0, 1 : seq_len + 1]

        # Mean pooling to obtain a single protein representation
        pooled = residue_embeddings.mean(dim=0)

        return pooled.cpu().numpy()
