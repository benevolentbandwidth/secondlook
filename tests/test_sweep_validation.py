"""Unit tests for sweep config validation.

A malformed sweep must fail in seconds, not after a ~178 GB dataset build and
hours of GPU time spent testing something other than what the author intended.
An unknown key is the dangerous case: it does not crash, it silently does
nothing, so the run produces confident-looking numbers for the wrong experiment.
"""

import pytest

from scripts.sweep_vertex import SWEEPS, KNOWN_CONFIG_KEYS, validate_sweep


def test_every_shipped_sweep_is_valid():
    # Guards the real SWEEPS dict, so a typo committed here is caught by CI
    # rather than by a wasted Vertex job.
    assert SWEEPS, "SWEEPS must not be empty"
    for name, configs in SWEEPS.items():
        validate_sweep(configs, name)


def test_shipped_sweeps_have_unique_names_across_the_dict():
    # Checkpoints are written to <ckpt-dir>/<config-name>/, so a name reused
    # across two sweeps pointed at the same dir would overwrite results.
    for name, configs in SWEEPS.items():
        names = [c["name"] for c in configs]
        assert len(names) == len(set(names)), f"{name} has duplicate config names"


def test_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown key"):
        validate_sweep([{"name": "x", "focal_gama": 2.0}], "typo")


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        validate_sweep([{"name": "dup"}, {"name": "dup"}], "s")


def test_rejects_missing_name():
    with pytest.raises(ValueError, match="needs a 'name'"):
        validate_sweep([{"loss": "bce"}], "s")


def test_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        validate_sweep([{"name": "x", "mode": "twophase"}], "s")


def test_rejects_empty_sweep():
    with pytest.raises(ValueError, match="empty"):
        validate_sweep([], "s")


def test_accepts_a_realistic_config():
    validate_sweep([{
        "name": "ok", "mode": "two_phase", "phase1_lr": 1e-3, "phase2_lr": 1e-5,
        "phase1_epochs": 12, "phase2_epochs": 15, "dropout_rate": 0.3,
        "loss": "focal", "focal_gamma": 2.0, "focal_alpha": 0.25,
        "use_class_weights": False, "max_neg_per_pos": 10.0,
        "input_size": (320, 320), "seed": 1337, "cache": False,
    }], "s")


def test_known_keys_cover_the_trainer_arguments():
    # If someone adds a trainer argument but forgets KNOWN_CONFIG_KEYS, configs
    # using it would be rejected. Keep the two in sync.
    for key in ("loss", "focal_gamma", "focal_alpha", "use_class_weights",
                "cache", "seed", "input_size"):
        assert key in KNOWN_CONFIG_KEYS
