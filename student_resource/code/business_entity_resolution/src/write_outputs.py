"""Write matching_results.tsv / candidate_pairs.tsv in the required format."""
import pandas as pd


def write_id_list_tsv(path: str, required_s1_ids, id_map: dict, id_col_name: str) -> None:
    """id_map: {source1_entity_id: iterable of S2/S3 ids}. Every required id gets a row."""
    rows = []
    for s1 in required_s1_ids:
        ids = sorted(set(id_map.get(s1, ())))
        rows.append((s1, ",".join(ids)))
    out = pd.DataFrame(rows, columns=["source1_entity_id", id_col_name])
    out.to_csv(path, sep="\t", index=False)


def pairs_to_id_map(pairs: pd.DataFrame, s1_col="source1_entity_id", id_col="cand_id") -> dict:
    return pairs.groupby(s1_col)[id_col].apply(set).to_dict()


def demo(tmp_path="/tmp/_wo_demo.tsv"):
    id_map = {"S1-1": {"S2-1", "S2-1", "S3-2"}, "S1-2": set()}
    write_id_list_tsv(tmp_path, ["S1-1", "S1-2", "S1-3"], id_map, "matched_entity_ids")
    df = pd.read_csv(tmp_path, sep="\t", dtype=str, keep_default_na=False)
    assert len(df) == 3
    row = df[df.source1_entity_id == "S1-1"].iloc[0]
    assert set(row.matched_entity_ids.split(",")) == {"S2-1", "S3-2"}
    row3 = df[df.source1_entity_id == "S1-3"].iloc[0]
    assert row3.matched_entity_ids == ""
    print("write_outputs.py demo OK")


if __name__ == "__main__":
    demo()
