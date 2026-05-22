"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_input_csv(tmp_path: Path) -> Path:
    """Create a 3-row input CSV with the canonical schema."""
    p = tmp_path / "input.csv"
    p.write_text(
        "base_id,original_scenario_a,variant_description,full_binary_prompt,virtue\n"
        'PRU-001,"orig A","variant 1","prompt text 1. Answer only with Yes or No.",Prudence\n'
        'PRU-001,"orig A","variant 2","prompt text 2. Answer only with Yes or No.",Prudence\n'
        'JUS-001,"orig B","variant 3","prompt text 3. Answer only with Yes or No.",Justice\n'
    )
    return p
