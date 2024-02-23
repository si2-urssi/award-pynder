#!/usr/bin/env python

from award_pynder.sources.nsf import NSF, NSF_PROGRAM_TO_CFDA_NUMBER_LUT, NSFPrograms

from ..utils import assert_dataset_basics

###############################################################################


def test_nsf() -> None:
    # Get data
    df = NSF.get_data(
        from_datetime="2014-01-01",
        to_datetime="2014-02-01",
        cfda_number=NSF_PROGRAM_TO_CFDA_NUMBER_LUT[NSFPrograms.Biological_Sciences],
        tqdm_kwargs={"leave": False},
    )

    # Run tests
    assert_dataset_basics(df)
