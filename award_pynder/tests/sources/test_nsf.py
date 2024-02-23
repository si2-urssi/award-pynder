#!/usr/bin/env python

from ..utils import assert_dataset_basics

from award_pynder.sources.nsf import NSF, NSFPrograms, NSF_PROGRAM_TO_CFDA_NUMBER_LUT

###############################################################################

def test_nsf() -> None:
    # Get data
    data = NSF.get_data(
        from_datetime="2014-01-01",
        to_datetime="2014-02-01",
        cfda_number=NSF_PROGRAM_TO_CFDA_NUMBER_LUT[NSFPrograms.Biological_Sciences],
        tqdm_kwargs={"leave": False},
    )

    # Run tests
    assert_dataset_basics(data)