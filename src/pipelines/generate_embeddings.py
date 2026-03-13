from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "antibody_developability_cleaned.csv"

CHAIN_CONFIG = {"vh": "vh_protein_sequence", "vl": "vl_protein_sequence"}

IDENTIFIER_COLS = ["antibody_id", "antibody_name"]


def generate_chain_embeddings(df, model_name, chain_name, sequence_col):
    """
    Generate embeddings for a single chain (VH or VL) using the specified model.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing antibody identifiers and sequence columns.
    model_name : str
        Key from MODEL_REGISTRY indicating which PLM wrapper to use.
    chain_name : str
        Short name for the chain, e.g. "vh" or "vl".
    sequence_col : str
        Name of the dataframe column containing sequences.

    Returns
    -------
    pd.DataFrame
        DataFrame containing identifier columns plus embedding columns.
    """

    model_class = MODEL_REGISTRY[model_name]
    model = model_class()

    embeddings = []

    for sequence in tqdm(
        df[sequence_col], desc=f"{model_name} {chain_name}", total=len(df)
    ):
        embeddings.append(model.embed(sequence))

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df = embeddings_df.add_prefix(f"{chain_name}_emb_")

    output_df = pd.concat(
        [df[IDENTIFIER_COLS].reset_index(drop=True), embeddings_df], axis=1
    )

    return output_df


def main():
    """
    Generate and cache VH/VL embeddings for all configured protein language models.

    If an embedding pickle file already exists, it is skipped to avoid
    unnecessary recomputation.
    """

    DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)

    for model_name in MODEL_REGISTRY:
        for chain_name, sequence_col in CHAIN_CONFIG.items():
            output_path = DATA_EMBEDDINGS_DIR / f"{model_name}_{chain_name}.pkl"

            if output_path.exists():
                print(f"[SKIP] {output_path.name} already exists")
                continue

            print(f"[RUN] Generating {model_name} {chain_name} embeddings...")
            embedding_df = generate_chain_embeddings(
                df=df,
                model_name=model_name,
                chain_name=chain_name,
                sequence_col=sequence_col,
            )
            embedding_df.to_pickle(output_path)
            print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
