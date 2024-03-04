#!/usr/bin/env python

from award_pynder.sources.nih import NIH

from ..utils import assert_dataset_basics

###############################################################################


def test_nih_basics() -> None:
    # Get data
    df = NIH.get_data(
        query="ethnography",
        from_datetime="2023-01-01",
        to_datetime="2023-06-01",
        tqdm_kwargs={"leave": False},
    )

    # Run tests
    assert_dataset_basics(df)


def test_nih_too_many_grants() -> None:
    # Get data
    try:
        NIH.get_data(
            query="software",
            from_datetime="2023-01-01",
            to_datetime="2023-02-01",
            tqdm_kwargs={"leave": False},
        )

    except ValueError as e:
        assert "too many" in str(e)
