#!/usr/bin/env python

from award_pynder.sources.sloan import Sloan

from ..utils import assert_dataset_basics

###############################################################################


def test_sloan() -> None:
    # Get data
    df = Sloan.get_data(
        query="software",
        from_datetime="2020-01-01",
        to_datetime="2022-01-01",
        tqdm_kwargs={"leave": False},
    )

    # Run tests
    assert_dataset_basics(df)
