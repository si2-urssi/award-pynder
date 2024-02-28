#!/usr/bin/env python

from award_pynder.sources.mellon import Mellon

from ..utils import assert_dataset_basics

###############################################################################


def test_mellon() -> None:
    # Get data
    df = Mellon.get_data(
        query="software",
        from_datetime="2023-01-01",
        to_datetime="2023-06-01",
        tqdm_kwargs={"leave": False},
    )

    # Run tests
    assert_dataset_basics(df)
