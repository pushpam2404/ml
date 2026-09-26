"""Loading raw source TSVs and assembling feature-ready pair frames."""
import pandas as pd
from normalize import prep_source

PREP_COLS = ["name_norm", "addr_norm", "pin", "house_no"]


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return prep_source(df)


def build_side_lookups(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame):
    """Build the two id-indexed lookup frames used to hydrate pair rows.

    Built once and reused across every pair chunk: an id-indexed frame lets pandas
    reuse its hash index on each join instead of re-hashing 10M candidate rows per
    chunk.
    """
    s1_side = s1.set_index("source1_entity_id")[PREP_COLS].add_suffix("_1")
    cand_side = pd.concat(
        [s2[["entity_id"] + PREP_COLS], s3[["entity_id"] + PREP_COLS]],
        ignore_index=True,
    ).set_index("entity_id")[PREP_COLS].add_suffix("_2")
    return s1_side, cand_side


def attach_pair_columns(pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame) -> pd.DataFrame:
    """Hydrate a pairs frame with both sides' prepped columns.

    ``pairs`` needs source1_entity_id and cand_id. Feed this one chunk at a time:
    the token-list columns make a fully-hydrated 100M-row pair frame far larger
    than memory.
    """
    return pairs.join(s1_side, on="source1_entity_id").join(cand_side, on="cand_id")
