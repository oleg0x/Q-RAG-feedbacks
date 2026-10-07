"""Vendored copies must match the read-only original while it is available.

The copies in ``src/`` make the reader/judge fork self-contained, but their
content is not ours: the prompts and the client belong to ``Q-RAG-feedback``.
While the original sits next to this repository, any drift in it must fail
these tests instead of accumulating silently. Where the original is absent
the tests are skipped and the copy itself is the contract.

The tail of each file is compared: only a header comment stating the copy's
origin is allowed above the original; below it the file matches byte for
byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
FEEDBACK_REPO = LAB.parent / "Q-RAG-feedback"

pytestmark = pytest.mark.skipif(
    not FEEDBACK_REPO.is_dir(),
    reason="../Q-RAG-feedback is not available; vendored copies are checked "
    "only where the original is present",
)

VENDORED = {
    "vendored_prompts.py": FEEDBACK_REPO / "prompts_and_metrics" / "prompts.py",
    "vendored_vllm_client.py": FEEDBACK_REPO / "vLLM_clients" / "sync_vllm_client.py",
}


@pytest.mark.parametrize("copy_name", sorted(VENDORED))
def test_vendored_copy_ends_with_the_original(copy_name: str) -> None:
    vendored = (LAB / "src" / copy_name).read_text(encoding="utf-8")
    original = VENDORED[copy_name].read_text(encoding="utf-8")
    assert vendored.endswith(original), (
        f"{copy_name} has diverged from the original {VENDORED[copy_name]}: "
        "update the copy to match it"
    )
