#!/usr/bin/env python

from __future__ import annotations
from typing import TYPE_CHECKING

from award_pynder.sources.base import ALL_DATASET_FIELDS

if TYPE_CHECKING:
    import pandas as pd

###############################################################################

def assert_dataset_basics(df: pd.DataFrame) -> None:
    # Assert that not only are all required fields present,
    # but that no extraneous fields are present as well
    assert set(df.columns) == set(ALL_DATASET_FIELDS)

    # Check no duplicates (overall)
    assert df.duplicated().sum() == 0

    # Check no duplicate ids
    assert df.id.nunique() == len(df)

    # Assert that there is at least some data
    assert len(df) > 0