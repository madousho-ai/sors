"""The new selector must load released v5.17 and reject stale version names."""
import pathlib
import random
import shutil
import tempfile

import pytest

from sors.data.synth_v5 import V5Rounds, load_synth_v5, parse_passes
from test_synth_v5 import _data
from test_train_cli import _mod


def test_v517_selector_normalizes_to_the_current_alias_and_rejects_duplicate_selection():
    assert _mod.parse_datasets('synth-v5.17') == ['synth-v5']
    with pytest.raises(SystemExit):
        _mod.parse_datasets('synth-v5+synth-v5.17')


def test_v516_selector_requires_the_pinned_old_checkout_instead_of_silently_reading_v517():
    with pytest.raises(SystemExit, match='synth-v5.17'):
        _mod.parse_datasets('synth-v5.16')


def test_default_loader_reads_v517_from_the_explicit_asset_root():
    fixture = _data()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            assets = pathlib.Path(tmp)
            shutil.copytree(fixture, assets / 'synth-intents-v5.17')
            (assets / 'synth-intents-v5.16').mkdir()
            (assets / 'synth-intents-v5.16/schema.py').write_text('raise AssertionError("stale asset selected")\n')
            items = load_synth_v5(datasets_dir=assets)
            assert len(items) == 10
            sampler = V5Rounds(items, parse_passes('breadth=1', {it.goal for it in items}))
            assert all(weight == 1 for weight in sampler.passes.values())
            assert len(sampler(8, random.Random(0))) == 8
    finally:
        shutil.rmtree(fixture)
