"""Loading raw source TSVs and assembling feature-ready pair frames."""
import pandas as pd
from normalize import prep_source

PREP_COLS = ["name_norm", "addr_norm", "pin", "house_no", "name_toks", "addr_toks"]


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return prep_source(df)


def attach_pair_columns(pairs: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Join the S1-side and candidate-side prepped columns onto a pairs frame.

    ``pairs`` needs source1_entity_id, cand_id, source (S2/S3).
    """
    s1_side = s1[["source1_entity_id"] + PREP_COLS].rename(
        columns={c: f"{c}_1" for c in PREP_COLS}
    )
    cand_all = pd.concat(
        [
            s2[["entity_id"] + PREP_COLS].rename(columns={"entity_id": "cand_id"}),
            s3[["entity_id"] + PREP_COLS].rename(columns={"entity_id": "cand_id"}),
        ],
        ignore_index=True,
    ).rename(columns={c: f"{c}_2" for c in PREP_COLS})

    out = pairs.merge(s1_side, on="source1_entity_id", how="left")
    out = out.merge(cand_all, on="cand_id", how="left")
    return out
