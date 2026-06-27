"""Unit tests for the checkpoint-path helper in modeling.train.

_checkpoint_path must keep forward slashes for remote (gs://) URIs even on
Windows, where os.path.join would insert a backslash and corrupt the URI.
"""

import os

from modeling.train import _checkpoint_path


def test_checkpoint_path_preserves_forward_slash_for_gcs():
    assert (
        _checkpoint_path("gs://b2-foundation/second-look/checkpoints/baseline", "best.keras")
        == "gs://b2-foundation/second-look/checkpoints/baseline/best.keras"
    )


def test_checkpoint_path_strips_trailing_slash_on_remote_uri():
    assert (
        _checkpoint_path("gs://bucket/dir/", "best.keras")
        == "gs://bucket/dir/best.keras"
    )


def test_checkpoint_path_never_emits_backslash_for_remote_uri():
    # Guards the Windows failure mode: a backslash anywhere breaks a gs:// URI.
    result = _checkpoint_path("gs://bucket/dir", "best.keras")
    assert "\\" not in result


def test_checkpoint_path_uses_os_join_for_local_paths():
    assert _checkpoint_path("checkpoints/baseline", "best.keras") == os.path.join(
        "checkpoints/baseline", "best.keras"
    )
