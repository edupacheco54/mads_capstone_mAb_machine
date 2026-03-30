from pathlib import Path
import pandas as pd
from src.models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
DATA_MODELING_DIR = PROJECT_ROOT / "data" / "modeling"
DATA_TEST_DIR = PROJECT_ROOT / "data" / "test_data"

CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "antibody_developability_cleaned.csv"
HOLDOUT_DATA_PATH = DATA_TEST_DIR / "cleaned_holdout_data.csv"

RAW_SEQUENCE_COLS = ["vh_protein_sequence", "vl_protein_sequence"]


def build_model_dataset(df_clean, model_name, embeddings_dir=None):
    """
    Build a model-ready dataset for a single protein language model.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataset containing identifiers, assay features, categorical
        features, raw sequence columns, and target.
    model_name : str
        Key from MODEL_REGISTRY indicating which model's embeddings to use.
    embeddings_dir : Path, optional
        Directory containing embedding pickle files. Defaults to DATA_EMBEDDINGS_DIR.

    Returns
    -------
    pd.DataFrame
        Model-ready dataframe containing original non-sequence features plus
        VH and VL embedding columns.
    """

    emb_dir = embeddings_dir if embeddings_dir is not None else DATA_EMBEDDINGS_DIR

    vh_path = emb_dir / f"{model_name}_vh.pkl"
    vl_path = emb_dir / f"{model_name}_vl.pkl"

    if not vh_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {vh_path}")
    if not vl_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {vl_path}")

    vh_df = pd.read_pickle(vh_path)
    vl_df = pd.read_pickle(vl_path)

    vl_feature_cols = [
        col for col in vl_df.columns if col not in ["antibody_id", "antibody_name"]
    ]

    df_model = df_clean.merge(vh_df, on=["antibody_id", "antibody_name"], how="left")
    df_model = df_model.merge(
        vl_df[["antibody_id", "antibody_name"] + vl_feature_cols],
        on=["antibody_id", "antibody_name"],
        how="left",
    )

    df_model = df_model.drop(columns=RAW_SEQUENCE_COLS)

    return df_model


def main():
    """
    Assemble and save model-ready datasets for each protein language model.

    Processes both the training data and the holdout data. The resulting files
    contain:
    - identifiers
    - assay features
    - categorical features
    - VH embeddings
    - VL embeddings
    - target (Titer)
    """

    DATA_MODELING_DIR.mkdir(parents=True, exist_ok=True)
    holdout_modeling_dir = DATA_MODELING_DIR / "holdout"
    holdout_modeling_dir.mkdir(parents=True, exist_ok=True)

    holdout_embeddings_dir = DATA_EMBEDDINGS_DIR / "holdout"

    datasets = [
        ("train", CLEANED_DATA_PATH, DATA_EMBEDDINGS_DIR, DATA_MODELING_DIR),
        ("holdout", HOLDOUT_DATA_PATH, holdout_embeddings_dir, holdout_modeling_dir),
    ]

    for dataset_name, data_path, embeddings_dir, output_dir in datasets:
        print(f"\n[DATASET] {dataset_name}")
        df_clean = pd.read_csv(data_path)

        for model_name in MODEL_REGISTRY:
            output_path = output_dir / f"{model_name}_model_df.pkl"

            print(f"[RUN] Building model-ready dataset for {model_name}...")
            df_model = build_model_dataset(
                df_clean=df_clean,
                model_name=model_name,
                embeddings_dir=embeddings_dir,
            )
            df_model.to_pickle(output_path)
            print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
