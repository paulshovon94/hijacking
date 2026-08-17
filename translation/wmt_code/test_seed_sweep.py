#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Self-tests for the split and voting logic in run_seed_sweep.py.

These run in seconds on synthetic data. They exist because the failure mode they guard
against is silent: a grouping bug would still produce plausible-looking accuracies, just
leaky ones, and nothing downstream would complain.

    python test_seed_sweep.py
"""

import os
import tempfile

import numpy as np
import pandas as pd

import run_seed_sweep as sweep


def _fake_groups(n_models: int = 216, rows_per_model: int = 96) -> np.ndarray:
    return np.repeat([f"./results/fam/size/{i}_cfg" for i in range(n_models)], rows_per_model)


def test_shadow_split_is_disjoint_and_sized() -> None:
    groups = _fake_groups()
    train_idx, val_idx = sweep.shadow_split(groups, seed=42)

    train_models = set(groups[train_idx])
    val_models = set(groups[val_idx])

    assert not (train_models & val_models), "train and val share a model"
    assert len(train_models) == 172, f"expected 172 train models, got {len(train_models)}"
    assert len(val_models) == 44, f"expected 44 val models, got {len(val_models)}"
    assert len(val_idx) == 44 * 96, f"expected 4224 val rows, got {len(val_idx)}"
    assert len(train_idx) + len(val_idx) == len(groups)
    print("shadow split: disjoint, 172/44 models, 4224 val rows")


def test_shadow_split_keeps_models_whole() -> None:
    """Every row of a model must land on the same side -- the point of the split."""
    groups = _fake_groups()
    train_idx, val_idx = sweep.shadow_split(groups, seed=32)
    for side in (train_idx, val_idx):
        counts = np.unique(groups[side], return_counts=True)[1]
        assert set(counts.tolist()) == {96}, f"a model was fragmented: {set(counts.tolist())}"
    print("shadow split: every model kept whole (96 rows on one side)")


def test_shadow_split_varies_with_seed() -> None:
    groups = _fake_groups()
    val_sets = []
    for seed in (32, 42, 52):
        _, val_idx = sweep.shadow_split(groups, seed=seed)
        val_sets.append(frozenset(groups[val_idx]))
    assert len(set(val_sets)) == 3, "different seeds produced identical holdouts"
    print("shadow split: three seeds give three different partitions")


def test_majority_vote_picks_the_mode() -> None:
    groups = np.array(["m1"] * 5 + ["m2"] * 5)
    preds = np.array([1, 1, 1, 0, 2] + [2, 2, 0, 2, 1])
    models, votes = sweep._majority_vote(preds, groups, n_classes=3)
    assert list(models) == ["m1", "m2"]
    assert list(votes) == [1, 2], f"unexpected votes: {votes}"
    print("majority vote: picks each model's modal prediction")


def test_vote_beats_noisy_rows() -> None:
    """The vote should recover the truth even when most individual rows are wrong."""
    groups = np.repeat(["m1", "m2", "m3"], 9)
    truth = np.repeat([0, 1, 2], 9)
    # 4 of 9 rows correct per model: minority per-row, but still the plurality.
    preds = np.array(
        [0, 0, 0, 0, 1, 1, 2, 2, 2] + [1, 1, 1, 1, 0, 0, 2, 2, 2] + [2, 2, 2, 2, 0, 0, 1, 1, 1]
    )
    scores = sweep.score_head(truth, preds, groups, n_classes=3, do_vote=True)
    assert abs(scores["row_acc"] - 4.0 / 9.0) < 1e-9, scores["row_acc"]
    assert scores["vote_acc"] == 1.0, scores["vote_acc"]
    print("vote: 0.444 per-row lifts to 1.000 per-model")


def test_vote_suppressed_without_group_split() -> None:
    groups = np.array(["m1"] * 4)
    scores = sweep.score_head(
        np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), groups, n_classes=2, do_vote=False
    )
    assert np.isnan(scores["vote_acc"]), "vote should be NaN for a row split"
    print("vote: correctly withheld when the split is row-level")


def test_inconsistent_model_labels_are_rejected() -> None:
    """A model's rows must all carry the same label; catch misalignment loudly."""
    groups = np.array(["m1"] * 4)
    try:
        sweep.score_head(
            np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), groups, n_classes=2, do_vote=True
        )
    except AssertionError:
        print("alignment: mismatched labels within a model raise, as intended")
        return
    raise AssertionError("expected an AssertionError for inconsistent labels")


def _write_csv(frame: pd.DataFrame) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    frame.to_csv(handle.name, index=False)
    handle.close()
    return handle.name


def test_load_groups_prefers_model_index() -> None:
    """Guards the bug that cost a 34-minute feature load: a wrong column name."""
    path = _write_csv(
        pd.DataFrame({"model_index": [0, 0, 1, 1], "model_dir": ["a", "a", "b", "b"],
                      "x1_file": ["f"] * 4})
    )
    try:
        groups = sweep.load_groups(path)
        assert list(groups) == [0, 0, 1, 1], groups
    finally:
        os.unlink(path)
    print("load_groups: resolves model_index from a real header")


def test_load_groups_falls_back_to_model_dir() -> None:
    path = _write_csv(pd.DataFrame({"model_dir": ["a", "a", "b"], "x1_file": ["f"] * 3}))
    try:
        groups = sweep.load_groups(path)
        assert list(groups) == ["a", "a", "b"], groups
    finally:
        os.unlink(path)
    print("load_groups: falls back to model_dir")


def test_load_groups_reports_available_columns() -> None:
    """A missing group column must name what *is* there, not just what is missing."""
    path = _write_csv(pd.DataFrame({"something_else": [1, 2]}))
    try:
        sweep.load_groups(path)
    except ValueError as exc:
        assert "something_else" in str(exc), f"error did not list actual columns: {exc}"
        print("load_groups: missing column error names the available columns")
        return
    finally:
        os.unlink(path)
    raise AssertionError("expected a ValueError for a missing group column")


def test_row_split_matches_existing_logic() -> None:
    """row_split must reproduce experiment_lora.main's split exactly."""
    n = 20736
    train_idx, val_idx = sweep.row_split(n, seed=42)
    rng = np.random.RandomState(42)
    expected = rng.permutation(n)
    assert np.array_equal(train_idx, expected[: int(0.8 * n)])
    assert np.array_equal(val_idx, expected[int(0.8 * n):])
    assert len(train_idx) == 16588 and len(val_idx) == 4148
    print("row split: byte-identical to the published protocol (16588/4148)")


if __name__ == "__main__":
    test_shadow_split_is_disjoint_and_sized()
    test_shadow_split_keeps_models_whole()
    test_shadow_split_varies_with_seed()
    test_majority_vote_picks_the_mode()
    test_vote_beats_noisy_rows()
    test_vote_suppressed_without_group_split()
    test_inconsistent_model_labels_are_rejected()
    test_load_groups_prefers_model_index()
    test_load_groups_falls_back_to_model_dir()
    test_load_groups_reports_available_columns()
    test_row_split_matches_existing_logic()
    print("\nAll split/vote self-tests passed.")
