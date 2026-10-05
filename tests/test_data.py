"""Leakage checks for the participant/family-level split, on synthetic data."""

import h5py
import numpy as np
import pandas as pd
import pytest

from src.data import build_leak_free_splits, check_leakage, load_and_split_data

VISITS = ("baseline", "followup_2y", "followup_4y")


def make_cohort(n_families=300, seed=0):
    """Families of 1-3 participants, each with up to 3 visits; ~10% flagged ADHD."""
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_families):
        for member in range(rng.integers(1, 4)):
            iid = f"P{fid:04d}_{member}"
            base_age = rng.uniform(9, 11)
            sex = int(rng.integers(1, 3))
            adhd = int(rng.random() < 0.1)
            for v, eid in enumerate(VISITS[: rng.integers(1, 4)]):
                rows.append({"iid": iid, "eid": eid, "fid": fid,
                             "age": base_age + 2 * v, "sex": sex, "ADHD1": adhd})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cohort():
    df = make_cohort()
    splits = build_leak_free_splits(
        df, df["age"].to_numpy(), df["sex"].to_numpy(),
        drop_eids=("followup_4y",), verbose=False,
    )
    return df, splits


def test_no_participant_or_family_overlap(cohort):
    df, splits = cohort
    assert check_leakage(df, splits)


def test_adhd_test_holds_every_visit_of_every_adhd_participant(cohort):
    df, splits = cohort
    adhd_rows = set(df.index[df["ADHD1"] == 1])
    assert adhd_rows == set(splits["adhd_test"])


def test_adhd_families_are_excluded_everywhere_else(cohort):
    df, splits = cohort
    adhd_fids = set(df.loc[df["ADHD1"] == 1, "fid"])
    for name in ("healthy_test", "train", "val"):
        assert not adhd_fids & set(df.loc[splits[name], "fid"]), name


def test_healthy_test_keeps_all_visits_of_selected_controls(cohort):
    df, splits = cohort
    healthy_iids = set(df.loc[splits["healthy_test"], "iid"])
    assert set(splits["healthy_test"]) == set(df.index[df["iid"].isin(healthy_iids)])
    assert (df.loc[splits["healthy_test"], "ADHD1"] == 0).all()


def test_dropped_visits_only_leave_train_and_val(cohort):
    df, splits = cohort
    for name in ("train", "val"):
        assert not (df.loc[splits[name], "eid"] == "followup_4y").any(), name


def test_split_is_reproducible():
    df = make_cohort()
    args = (df, df["age"].to_numpy(), df["sex"].to_numpy())
    assert (build_leak_free_splits(*args, verbose=False)
            == build_leak_free_splits(*args, verbose=False))


def test_load_and_split_data_end_to_end(tmp_path):
    df = make_cohort(n_families=120)
    n = len(df)
    mat_path, npy_path = tmp_path / "cohort.mat", tmp_path / "cohort.npy"
    with h5py.File(mat_path, "w") as f:
        f["iid"] = df["iid"].to_numpy().astype("S")
        f["eid"] = df["eid"].to_numpy().astype("S")
        f["fid"] = df["fid"].to_numpy(dtype=float)
        X = np.zeros((7, n))
        X[0], X[2], X[3] = df["age"], df["sex"], df["ADHD1"]
        f["X"] = X
    np.save(npy_path, np.zeros((n, 4, 4, 4), dtype=np.float32))

    train, val, adhd_test, healthy_test, *demogs = load_and_split_data(
        str(mat_path), str(npy_path), drop_eids=())
    assert len(train) + len(val) + len(adhd_test) + len(healthy_test) <= n
    assert train[0][0].shape == (1, 4, 4, 4)
    assert all(len(d) == len(s) for d, s in zip(demogs, (train, val, adhd_test, healthy_test)))
