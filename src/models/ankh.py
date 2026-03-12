import torch
import ankh

from src.utils.device import get_best_device


class AnkhModel:
    """
    Wrapper for generating protein embeddings using the ANKH protein language model.

    This class loads a pretrained ANKH model (base or large) and provides a
    simple interface to convert an amino acid sequence into a fixed-length
    embedding vector. The model produces per-residue embeddings, which are
    aggregated using mean pooling to obtain a single representation for the
    entire protein sequence.

    Parameters
    ----------
    model_size : str, optional
        Size of the ANKH model to load. Options include:
        - "base"  : smaller model with lower memory usage
        - "large" : larger model with higher capacity
    device : str, optional
        PyTorch device identifier ("cuda", "mps", or "cpu"). If not provided,
        the best available device is selected automatically using
        `get_best_device()`.

    Notes
    -----
    ANKH tokenizes amino acid sequences at the residue level. Each amino acid
    is treated as an individual token, so sequences are converted into lists
    of characters before tokenization.

    The resulting per-residue embeddings are pooled (averaged) to produce
    a single embedding vector suitable for downstream machine learning tasks.
    """

    def __init__(self, model_size="base", device=None):
        """
        Initialize the ANKH embedding model.

        Loads the specified pretrained ANKH model and moves it to the chosen
        computation device.

        Parameters
        ----------
        model_size : str
            Size of the ANKH model to load ("base" or "large").
        device : str or None
            Device to run the model on. If None, the best available device
            is automatically selected.
        """

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
        """
        Generate a fixed-length embedding for a protein sequence.

        The input sequence is tokenized into individual amino acids and passed
        through the ANKH model to obtain per-residue embeddings. Padding and
        special tokens are removed using the attention mask, and the remaining
        residue embeddings are mean-pooled to produce a single vector
        representation for the entire protein.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array representing the pooled protein embedding.
            The dimensionality depends on the ANKH model size used.
        """

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

        # Per-residue embeddings for the sequence
        embeddings = embedding_repr.last_hidden_state[0]

        # Remove padding and special tokens using the attention mask
        mask = attention_mask[0].bool()
        residue_embeddings = embeddings[mask]

        # Mean pooling to obtain a single protein representation
        pooled = residue_embeddings.mean(dim=0)

        return pooled.cpu().numpy()
