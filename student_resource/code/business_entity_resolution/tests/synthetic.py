"""Tiny made-up dataset in the challenge's file layout, for smoke tests.

Nothing here comes from the real challenge data. It only has to be shaped like it:
three tab-separated sources, a ground-truth file, a country column, and the noise
types the problem statement lists (suffix variants, word order, typos, abbreviations,
missing PIN). Seeded, so a run is repeatable.
"""
import csv
import os
import random

SYL = ["ka", "ri", "to", "mu", "sen", "dor", "val", "ex", "nia", "bel", "tan", "zor",
       "pel", "gan", "lu", "mer", "vis", "hon", "dra", "fel", "qui", "jax", "wen", "bro"]
SUFFIX_FORMS = [("Pvt Ltd", "Private Limited"), ("Corp", "Corporation"),
                ("Inc", "Incorporated"), ("Co", "Company"), ("LLC", "LLC")]
STREETS = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Lane", "Ln")]


def _word(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3)))


def _typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]  # drop one letter


def _make_entity(rng, country):
    name = [_word(rng).title(), _word(rng).title()]
    sfx = rng.choice(SUFFIX_FORMS)
    street = rng.choice(STREETS)
    pin = str(rng.randint(10000, 99999))
    num = rng.randint(1, 400)
    addr_words = [_word(rng).title(), street[0], _word(rng).title()]
    return {"name": name, "sfx": sfx, "street": street, "num": num, "addr_words": addr_words,
            "pin": pin, "country": country}


def _canonical(e):
    name = " ".join(e["name"] + [e["sfx"][0]])
    addr = f"{e['num']} {e['addr_words'][0]} {e['street'][0]}, {e['addr_words'][2]} {e['pin']}"
    return name, addr


def _noisy(e, rng):
    name = list(e["name"])
    if rng.random() < 0.3:
        name.reverse()
    if rng.random() < 0.3:
        name[0] = _typo(name[0], rng)
    name = " ".join(name + [e["sfx"][rng.randint(0, 1)]])
    if rng.random() < 0.3:
        name = name.upper()
    street = e["street"][rng.randint(0, 1)]
    pin = e["pin"] if rng.random() < 0.7 else ""
    num = e["num"] if rng.random() < 0.85 else e["num"] + 1
    addr = f"{num} {e['addr_words'][0]} {street}, {e['addr_words'][2]} {pin}".strip()
    if rng.random() < 0.05:
        addr = ""
    return name, addr


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def make_split(out_dir, prefix, n_s1, countries, seed, with_truth):
    rng = random.Random(seed)
    s1, s2, s3, truth = [], [], [], []
    n2 = n3 = 0
    for i in range(n_s1):
        country = countries[i % len(countries)]
        e = _make_entity(rng, country)
        name, addr = _canonical(e)
        sid = f"S1-{i + 1:05d}"
        s1.append((sid, name, addr, country))
        matches = []
        n_match = 0 if rng.random() < 0.06 else rng.randint(1, 4)
        for _ in range(n_match):
            n, a = _noisy(e, rng)
            if rng.random() < 0.5:
                n2 += 1
                cid = f"S2-{n2:05d}"
                s2.append((cid, n, a, country))
            else:
                n3 += 1
                cid = f"S3-{n3:05d}"
                s3.append((cid, n, a, country))
            matches.append(cid)
        truth.append((sid, ",".join(matches)))
    # unrelated records that should never match anything
    for _ in range(n_s1):
        country = countries[rng.randrange(len(countries))]
        e = _make_entity(rng, country)
        n, a = _noisy(e, rng)
        n2 += 1
        s2.append((f"S2-{n2:05d}", n, a, country))
    os.makedirs(out_dir, exist_ok=True)
    hdr = ["entity_id", "business_name", "business_address", "country"]
    _write(os.path.join(out_dir, f"{prefix}_source1.tsv"), hdr, s1)
    _write(os.path.join(out_dir, f"{prefix}_source2.tsv"), hdr, s2)
    _write(os.path.join(out_dir, f"{prefix}_source3.tsv"), hdr, s3)
    if with_truth:
        _write(os.path.join(out_dir, f"{prefix}_ground_truth.tsv"),
               ["source1_entity_id", "matched_entity_ids"], truth)


def make_dataset(root, n_train=1500, n_test=500):
    """root/train/train_*.tsv and root/test/test_*.tsv. The test set adds a country the
    training set never saw, as in the challenge."""
    make_split(os.path.join(root, "train"), "train", n_train, ["US", "India"], 1, True)
    make_split(os.path.join(root, "test"), "test", n_test, ["US", "India", "France"], 2, False)
