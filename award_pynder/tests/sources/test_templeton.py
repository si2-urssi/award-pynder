#!/usr/bin/env python

from award_pynder.sources.templeton import Templeton

from ..utils import assert_dataset_basics

###############################################################################


def test_templeton() -> None:
    # Get data
    df = Templeton.get_data(
        query="software",
        from_datetime="2020-01-01",
        to_datetime="2022-01-01",
    )

    # Run tests
    assert_dataset_basics(df)
