"""
Data loading and participant-level train/val/test splitting for the brain-age model.

The split logic here replaced an earlier version that matched healthy controls
at a single timepoint and removed them from the train/validation pool by row
index. That earlier approach leaked data: a control's OTHER visits (same
participant, different timepoint) kept the same participant id and stayed in
the training pool, so the model could see the same brain in both training and
test data. This version excludes data by PARTICIPANT and by FAMILY, not by
row, and reports the overlap between splits so the absence of leakage can be
verified rather than assumed.

Split design:
  1. ADHD test set   = every row (all visits) of every participant flagged
                        ADHD at any timepoint.
  2. Every family containing an ADHD participant is removed from everything
     else, so relatives of ADHD participants cannot appear in training.
  3. Healthy test set = age- and sex-matched controls, drawn only from the
     ADHD-free pool, sized as a multiple of the ADHD participant count.
  4. Train / validation = the remaining participants, split by family.
"""

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import TensorDataset, Subset


def dereference_strings(mat_file_path, raw):
    """
    Resolve a MATLAB v7.3 cell array of strings read via h5py into Python strings.

    Each cell in a v7.3 .mat file is stored as its own HDF5 dataset, so h5py
    hands back object references instead of text. Two references to the same
    string do not compare as equal, so grouping by participant id silently
    breaks unless the references are dereferenced first.
    """
    if raw.dtype == np.object_ or "reference" in str(raw.dtype):
        with h5py.File(mat_file_path, "r") as mat_file:
            values = []
            for ref in raw.flatten():
                obj = mat_file[ref]
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                elif isinstance(val, np.ndarray):
                    val = "".join(chr(c) for c in val.flatten())
                values.append(str(val).strip())
            return np.array(values)
    if isinstance(raw.flatten()[0], bytes):
        return np.array([e.decode("utf-8") for e in raw.flatten()])
    return np.array([str(e) for e in raw.flatten()])


def check_leakage(df, splits, keys=("iid", "fid")):
    """Print the pairwise overlap between splits at the participant and family level."""
    names = list(splits.keys())
    clean = True
    print(f"\n{'=' * 70}")
    print("LEAKAGE REPORT")
    print(f"{'=' * 70}")
    for key in keys:
        print(f"\n--- overlap by {key} ---")
        sets = {name: set(df.loc[idx, key]) for name, idx in splits.items()}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = len(sets[a] & sets[b])
                clean &= overlap == 0
                tag = "" if overlap == 0 else "   <-- LEAK"
                print(f"{a:>13} n {b:<13}: {overlap:>6} unique {key}s{tag}")
    assigned = sum(len(v) for v in splits.values())
    print(f"\nrows assigned: {assigned} / {len(df)}   "
          f"({len(df) - assigned} deliberately dropped)")
    print("CLEAN" if clean else "LEAKAGE PRESENT")
    print(f"{'=' * 70}")
    return clean


def build_leak_free_splits(
    df,
    age,
    sex,
    adhd_category="ADHD1",
    healthy_ratio=3,
    age_bins=15,
    exclude_healthy_families=True,
    drop_eids=("8_year_follow_up_y_arm_1",),
    train_frac=0.8,
    split_by="fid",
    random_state=42,
    verbose=True,
):
    """
    Build the four-way split. `df` must have a 0..N-1 RangeIndex aligned
    positionally with `age` and `sex`.
    """
    rng = np.random.default_rng(random_state)

    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("df needs a 0..N-1 RangeIndex aligned with age/sex.")

    def log(msg=""):
        if verbose:
            print(msg)

    # 1. ADHD test set: every row of every participant flagged ADHD at any timepoint.
    flag = df[adhd_category] == 1
    adhd_iids = set(df.loc[flag, "iid"].unique())
    adhd_test = df.index[df["iid"].isin(adhd_iids)].to_numpy()

    log(f"\n{'=' * 70}")
    log("1. ADHD TEST SET")
    log(f"{'=' * 70}")
    log(f"ADHD participants: {len(adhd_iids)}   rows (all timepoints): {len(adhd_test)}")

    # 2. Block every family that contains an ADHD participant.
    adhd_fids = set(df.loc[df["iid"].isin(adhd_iids), "fid"].unique())
    blocked = set(df.index[df["fid"].isin(adhd_fids)].tolist())

    log(f"\n{'=' * 70}")
    log("2. ADHD FAMILY EXCLUSION")
    log(f"{'=' * 70}")
    log(f"Families touched by ADHD: {len(adhd_fids)}   rows blocked: {len(blocked)}")

    # 3. Age- and sex-matched healthy controls, drawn from the ADHD-free pool.
    flagged_rows = df.index[flag].to_numpy()
    reference = (
        pd.DataFrame(
            {"iid": df.loc[flagged_rows, "iid"].to_numpy(),
             "age": age[flagged_rows],
             "sex": sex[flagged_rows]},
            index=flagged_rows,
        )
        .sort_index()
        .drop_duplicates("iid", keep="first")
    )

    n_adhd_participants = len(reference)
    target_total = int(round(healthy_ratio * n_adhd_participants))

    bin_edges = np.histogram_bin_edges(reference["age"].to_numpy(), bins=age_bins)
    reference_bin = np.digitize(reference["age"].to_numpy(), bin_edges)

    strata, strata_counts = np.unique(
        np.column_stack([reference_bin, reference["sex"].to_numpy()]), axis=0, return_counts=True
    )
    target_counts = {
        (int(b), int(s)): int(round(c / n_adhd_participants * target_total))
        for (b, s), c in zip(strata, strata_counts)
    }
    drift = target_total - sum(target_counts.values())
    if drift:
        biggest = max(target_counts, key=target_counts.get)
        target_counts[biggest] += drift

    candidate_rows = df.index[(~df.index.isin(blocked)) & (df[adhd_category] == 0)].to_numpy()
    candidates = pd.DataFrame(
        {"iid": df.loc[candidate_rows, "iid"].to_numpy(),
         "age": age[candidate_rows],
         "sex": sex[candidate_rows]},
        index=candidate_rows,
    )
    candidates["bin"] = np.digitize(candidates["age"].to_numpy(), bin_edges)

    log(f"\n{'=' * 70}")
    log("3. HEALTHY TEST SET (age + sex matched)")
    log(f"{'=' * 70}")
    log(f"ADHD participants to match: {n_adhd_participants}   "
        f"target healthy participants: {target_total} ({healthy_ratio}x)")

    chosen_rows, used_iids, shortfall = [], set(), 0
    for (b, s), need in sorted(target_counts.items()):
        pool = candidates[(candidates["bin"] == b) & (candidates["sex"] == s)
                           & (~candidates["iid"].isin(used_iids))]
        pool = pool.drop_duplicates("iid", keep="first")
        take = min(need, len(pool))
        if take:
            picked = pool.index.to_numpy()[rng.choice(len(pool), size=take, replace=False)]
            chosen_rows.extend(picked.tolist())
            used_iids.update(pool.loc[picked, "iid"].tolist())
        shortfall += need - take

    if shortfall:
        log(f"Short by {shortfall} participants; those age/sex strata are exhausted.")

    healthy_iids = set(df.loc[chosen_rows, "iid"])
    healthy_test = df.index[df["iid"].isin(healthy_iids)].to_numpy()

    log(f"Healthy participants selected: {len(healthy_iids)}   "
        f"test rows (all timepoints): {len(healthy_test)}")

    # 4. Block the selected controls (and optionally their families) from train/val.
    if exclude_healthy_families:
        healthy_fids = set(df.loc[df["iid"].isin(healthy_iids), "fid"].unique())
        healthy_blocked = set(df.index[df["fid"].isin(healthy_fids)].tolist())
    else:
        healthy_blocked = set(df.index[df["iid"].isin(healthy_iids)].tolist())
    blocked |= healthy_blocked

    # 5. Train / validation over the survivors, split by family so no family
    #    appears in both.
    pool_mask = ~df.index.isin(blocked)
    if drop_eids:
        pool_mask &= ~df["eid"].isin(list(drop_eids))
    pool = df[pool_mask]

    keys = pool[split_by].unique()
    keys = keys[rng.permutation(len(keys))]
    n_train = int(len(keys) * train_frac)
    train_keys, val_keys = set(keys[:n_train]), set(keys[n_train:])
    train = pool.index[pool[split_by].isin(train_keys)].tolist()
    val = pool.index[pool[split_by].isin(val_keys)].tolist()

    log(f"\n{'=' * 70}")
    log("4. TRAIN / VALIDATION")
    log(f"{'=' * 70}")
    log(f"Survivors: {len(pool)} rows / {pool['fid'].nunique()} families")
    log(f"Train: {len(train):>7} rows ({len(train) / max(len(pool), 1) * 100:5.1f}%)")
    log(f"Val:   {len(val):>7} rows ({len(val) / max(len(pool), 1) * 100:5.1f}%)")

    return {
        "adhd_test": adhd_test.tolist(),
        "healthy_test": healthy_test.tolist(),
        "train": train,
        "val": val,
    }


def load_and_split_data(
    mat_file_path,
    npy_file_path,
    train_percentage=0.80,
    val_percentage=0.20,
    adhd_category="ADHD1",
    healthy_ratio=3,
    age_bins=15,
    exclude_healthy_families=True,
    drop_eids=("8_year_follow_up_y_arm_1",),
    split_by="fid",
    random_state=42,
):
    """
    Load the T1 brain volumes and demographics, then build the leak-free
    train / validation / ADHD-test / healthy-test split described above.

    Returns four `TensorDataset` subsets, their index arrays, and four
    demographics dataframes (one per split), in that order.
    """
    print("Loading images and demographics...")
    images = np.load(npy_file_path)
    with h5py.File(mat_file_path, "r") as mat_file:
        iid_raw = mat_file["iid"][:]
        eid_raw = mat_file["eid"][:]
        fid = mat_file["fid"][:].flatten()
        data = mat_file["X"][:]

    age = data[0, :].flatten()
    sex = data[2, :].flatten().astype(int)
    adhd_columns = {"ADHD1": data[3, :], "ADHD2": data[4, :],
                     "ADHD3": data[5, :], "ADHD4": data[6, :]}
    if adhd_category not in adhd_columns:
        raise ValueError(f"Invalid ADHD category: {adhd_category}. "
                          f"Must be one of {list(adhd_columns)}")
    adhd_flag = adhd_columns[adhd_category].flatten()

    iid = dereference_strings(mat_file_path, iid_raw)
    eid = dereference_strings(mat_file_path, eid_raw)

    df = pd.DataFrame({
        "iid": iid, "eid": eid, "fid": fid, "age": age, "sex": sex,
        adhd_category: adhd_flag,
    }).reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['iid'].nunique()} participants, "
          f"{df['fid'].nunique()} families, {len(df['eid'].unique())} timepoints")

    print("\nBuilding leak-free train/val/test split...")
    train_frac = train_percentage / (train_percentage + val_percentage)
    splits = build_leak_free_splits(
        df, age, sex,
        adhd_category=adhd_category,
        healthy_ratio=healthy_ratio,
        age_bins=age_bins,
        exclude_healthy_families=exclude_healthy_families,
        drop_eids=drop_eids,
        train_frac=train_frac,
        split_by=split_by,
        random_state=random_state,
    )
    check_leakage(df, splits)

    images_tensor = torch.tensor(np.expand_dims(images, axis=1), dtype=torch.float32)
    age_tensor = torch.tensor(age, dtype=torch.float32)
    dataset = TensorDataset(images_tensor, age_tensor)

    def demographics_for(indices):
        idx = np.asarray(indices, dtype=int)
        demog = df.loc[idx, ["iid", "eid", "fid", "age", "sex", adhd_category]].reset_index(drop=True)
        demog.insert(0, "index", idx)
        return demog

    train_indices, val_indices = splits["train"], splits["val"]
    adhd_test_indices, healthy_test_indices = splits["adhd_test"], splits["healthy_test"]

    print(f"\nFinal dataset sizes: train {len(train_indices)} | val {len(val_indices)} | "
          f"adhd_test {len(adhd_test_indices)} | healthy_test {len(healthy_test_indices)}\n")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, adhd_test_indices),
        Subset(dataset, healthy_test_indices),
        demographics_for(train_indices),
        demographics_for(val_indices),
        demographics_for(adhd_test_indices),
        demographics_for(healthy_test_indices),
    )
