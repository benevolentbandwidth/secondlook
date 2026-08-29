"""Unit tests for the class-imbalance levers: focal loss, train-mix
rebalancing, and temperature-scaling calibration.

These are the three knobs added to attack the full-run weak point (WORTH
precision 0.15-0.19, specificity ~0.56 at the 0.80 sensitivity floor). They run
on CPU in milliseconds -- no GPU, no dataset, no GCS -- so a broken lever is
caught before a Vertex job spends hours proving it.

The safety-critical assertions here are:
  * unknown config values RAISE rather than silently defaulting (CLAUDE.md
    failure mode #2),
  * mix rebalancing NEVER drops a WORTH_SECOND_LOOK case (failure mode #1),
  * temperature scaling is strictly monotonic, so it cannot move the operating
    point chosen at the sensitivity floor.
"""

import numpy as np
import pandas as pd
import pytest

from modeling.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    probabilities_to_logits,
)
from modeling.dataset_mix import rebalance_train_mix, train_mix_stats
from modeling.losses import build_loss, describe_loss


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def test_build_loss_defaults_to_binary_crossentropy():
    # The validated baseline must remain the default: adding the focal option
    # must not silently change what an existing caller trains with.
    assert build_loss() == "binary_crossentropy"
    assert build_loss("bce") == "binary_crossentropy"


def test_build_loss_returns_focal_object_with_gamma():
    loss = build_loss("focal", focal_gamma=3.0)
    assert loss.gamma == 3.0
    # alpha=None must leave focal's internal class balancing OFF, so the
    # imbalance is not counted twice alongside Keras class_weight.
    assert loss.apply_class_balancing is False


def test_build_loss_focal_alpha_enables_class_balancing():
    loss = build_loss("focal", focal_gamma=2.0, focal_alpha=0.25)
    assert loss.apply_class_balancing is True
    assert loss.alpha == 0.25


def test_build_loss_rejects_unknown_loss_name():
    # No silent fallback to a default loss -- a typo in SWEEP_CONFIGS must not
    # quietly train the wrong objective for hours on a GPU.
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("focal_loss")


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.5])
def test_build_loss_rejects_out_of_range_alpha(bad_alpha):
    with pytest.raises(ValueError, match="focal_alpha"):
        build_loss("focal", focal_alpha=bad_alpha)


def test_build_loss_rejects_negative_gamma():
    with pytest.raises(ValueError, match="focal_gamma"):
        build_loss("focal", focal_gamma=-1.0)


def test_describe_loss_names_the_class_weight_state():
    assert "class_weight=OFF" in describe_loss("focal", 2.0, 0.25, False)
    assert "class_weight=on" in describe_loss("bce", 2.0, None, True)


# ---------------------------------------------------------------------------
# Train-mix rebalancing
# ---------------------------------------------------------------------------

def _mix_frame() -> pd.DataFrame:
    """A miniature of the real combined train split: RSNA swamps the others and
    is almost all negative; CBIS is small and mostly positive."""
    rows = []
    rows += [{"dataset": "rsna", "canonical_label": 1}] * 10
    rows += [{"dataset": "rsna", "canonical_label": 0}] * 490
    rows += [{"dataset": "cbis", "canonical_label": 1}] * 40
    rows += [{"dataset": "cbis", "canonical_label": 0}] * 10
    rows += [{"dataset": "vindr", "canonical_label": 1}] * 5
    rows += [{"dataset": "vindr", "canonical_label": 0}] * 95
    return pd.DataFrame(rows)


def test_rebalance_is_a_noop_without_knobs():
    df = _mix_frame()
    out = rebalance_train_mix(df, verbose=False)
    assert out is df  # unchanged object: default behavior is untouched


def test_rebalance_never_drops_positives():
    # The #1 failure mode is false reassurance. Rebalancing may only ever remove
    # negatives; losing a WORTH case to a mix knob would be a safety regression.
    df = _mix_frame()
    out = rebalance_train_mix(df, max_neg_per_pos=1.0, verbose=False)
    before = df[df["canonical_label"] == 1].groupby("dataset").size()
    after = out[out["canonical_label"] == 1].groupby("dataset").size()
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_rebalance_caps_negatives_per_dataset_independently():
    out = rebalance_train_mix(_mix_frame(), max_neg_per_pos=2.0, verbose=False)
    counts = out.groupby(["dataset", "canonical_label"]).size().unstack(fill_value=0)
    # rsna: 10 positives -> at most 20 negatives (was 490).
    assert counts.loc["rsna", 0] == 20
    # cbis: 40 positives -> cap of 80, but it only HAS 10 negatives. A cap must
    # never invent or over-trim rows it does not apply to.
    assert counts.loc["cbis", 0] == 10
    assert counts.loc["vindr", 0] == 10


def test_rebalance_raises_the_positive_rate():
    df = _mix_frame()
    before = train_mix_stats(df)["train_pos_rate"]
    after = train_mix_stats(
        rebalance_train_mix(df, max_neg_per_pos=2.0, verbose=False)
    )["train_pos_rate"]
    assert after > before


def test_dataset_fractions_subsample_only_named_datasets():
    out = rebalance_train_mix(
        _mix_frame(), dataset_fractions={"rsna": 0.1}, verbose=False
    )
    counts = out.groupby(["dataset", "canonical_label"]).size().unstack(fill_value=0)
    assert counts.loc["rsna", 0] == 49       # 10% of 490
    assert counts.loc["rsna", 1] == 10       # positives untouched
    assert counts.loc["vindr", 0] == 95      # not named -> unchanged
    assert counts.loc["cbis", 0] == 10


def test_rebalance_is_deterministic_under_the_fixed_seed():
    a = rebalance_train_mix(_mix_frame(), max_neg_per_pos=2.0, verbose=False)
    b = rebalance_train_mix(_mix_frame(), max_neg_per_pos=2.0, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_rebalance_rejects_unknown_dataset_name():
    # Catches a typo in SWEEP_CONFIGS before it silently rebalances nothing.
    with pytest.raises(ValueError, match="not in the train split"):
        rebalance_train_mix(_mix_frame(), dataset_fractions={"rnsa": 0.5})


def test_rebalance_rejects_non_binary_labels():
    df = _mix_frame()
    df.loc[0, "canonical_label"] = 2
    with pytest.raises(ValueError, match="Unknown label values"):
        rebalance_train_mix(df, max_neg_per_pos=2.0)


def test_rebalance_rejects_missing_column():
    df = _mix_frame().drop(columns=["dataset"])
    with pytest.raises(ValueError, match="needs column"):
        rebalance_train_mix(df, max_neg_per_pos=2.0)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def _overconfident_probabilities(seed: int = 42, n: int = 20000):
    """Labels plus (a) perfectly calibrated and (b) deliberately over-confident
    probabilities -- the latter being the failure mode temperature scaling undoes.

    Labels are drawn FROM the calibrated probabilities, which is what makes them
    calibrated: p really is P(y=1 | p). Drawing labels first and then picking
    logits per label would NOT produce calibrated probabilities, because the
    class prior shifts the posterior's intercept, and temperature scaling is
    scale-only (no bias term) and could not recover it.

    The logit mean is set so the positive rate lands near the real ~6%.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(-3.0, 1.5, size=n)
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    labels = rng.binomial(1, calibrated)
    overconfident = 1.0 / (1.0 + np.exp(-logits * 3.0))
    return labels, calibrated, overconfident


def test_calibration_fixture_is_actually_calibrated_and_imbalanced():
    # Guards the tests below: if the fixture drifts, their failures would be
    # meaningless. Observed rate must track predicted probability, at ~6% positive.
    labels, calibrated, _ = _overconfident_probabilities()
    assert 0.03 < labels.mean() < 0.12
    assert expected_calibration_error(labels, calibrated) < 0.02


def test_apply_temperature_of_one_is_a_noop():
    _, _, probs = _overconfident_probabilities()
    np.testing.assert_allclose(apply_temperature(probs, 1.0), probs, atol=1e-6)


def test_apply_temperature_is_monotonic():
    # Monotonicity is the whole safety argument: a non-decreasing map leaves the
    # ROC curve -- and so the operating point at the 0.80 sensitivity floor --
    # unchanged, meaning calibration can never cost WORTH sensitivity.
    # Non-DECREASING, not strictly increasing: _EPS clipping merges scores in the
    # far tail that were already numerically indistinguishable.
    _, _, probs = _overconfident_probabilities()
    order = np.argsort(probs, kind="stable")
    ascending = probs[order]
    for temperature in (0.5, 2.0, 7.5):
        scaled = apply_temperature(probs, temperature)[order]
        assert np.all(np.diff(scaled) >= 0)
        # And ties are only ever created, never reordered: wherever the input
        # was strictly increasing outside the clipped tail, the output is too.
        interior = (ascending[:-1] > 1e-9) & (ascending[1:] < 1 - 1e-9)
        assert np.all(np.diff(scaled)[interior] > 0)


def test_apply_temperature_preserves_auroc():
    # The direct statement of what monotonicity buys us, on the metric we report.
    from sklearn.metrics import roc_auc_score

    labels, _, probs = _overconfident_probabilities()
    before = roc_auc_score(labels, probs)
    for temperature in (0.5, 2.0, 7.5):
        after = roc_auc_score(labels, apply_temperature(probs, temperature))
        assert after == pytest.approx(before, abs=1e-6)


def test_fit_temperature_corrects_overconfidence():
    labels, _, overconfident = _overconfident_probabilities()
    temperature = fit_temperature(labels, overconfident, verbose=False)
    # Over-confident inputs need T > 1 to be softened back toward the base rate.
    assert temperature > 1.0
    before = expected_calibration_error(labels, overconfident)
    after = expected_calibration_error(
        labels, apply_temperature(overconfident, temperature)
    )
    assert after < before


def test_fit_temperature_recovers_the_known_scale_factor():
    # Probabilities were built by tripling the logits, so the fit should land
    # near T = 3 -- a real check that the optimizer works, not just that ECE dropped.
    labels, _, overconfident = _overconfident_probabilities()
    temperature = fit_temperature(labels, overconfident, verbose=False)
    assert 2.5 < temperature < 3.5


def test_fit_temperature_leaves_calibrated_probabilities_alone():
    labels, calibrated, _ = _overconfident_probabilities()
    temperature = fit_temperature(labels, calibrated, verbose=False)
    assert 0.85 < temperature < 1.15


def test_fit_temperature_rejects_single_class_split():
    # Silently returning T=1 here would report an uncalibrated model as calibrated.
    probs = np.linspace(0.1, 0.9, 50)
    with pytest.raises(ValueError, match="single class"):
        fit_temperature(np.zeros(50, dtype=int), probs)


def test_fit_temperature_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        fit_temperature(np.array([0, 1, 0]), np.array([0.2, 0.8]))


def test_fit_temperature_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        fit_temperature(np.array([]), np.array([]))


def test_apply_temperature_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="must be > 0"):
        apply_temperature(np.array([0.5]), 0.0)


def test_logit_transform_is_finite_at_saturation():
    # This model produces saturated sigmoid outputs; an inf here would poison
    # the whole calibration fit.
    logits = probabilities_to_logits(np.array([0.0, 1.0, 0.5]))
    assert np.all(np.isfinite(logits))
