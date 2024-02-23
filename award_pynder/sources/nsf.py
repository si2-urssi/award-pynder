#!/usr/bin/env python

from __future__ import annotations
from datetime import datetime

from .base import DataSource, DatasetFields, ALL_DATASET_FIELDS

import requests
import pandas as pd
from tqdm import tqdm

import logging

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

_DEFAULT_METADATA_SET = [
    "id",
    "date",
    "startDate",
    "expDate",
    "title",
    "awardeeName",
    "piFirstName",
    "piLastName",
    "cfdaNumber",
    "estimatedTotalAmt",
    "abstractText",
    "piEmail",
    # "fundProgramCode",
    # "awardeeCounty",
]

_DEFAULT_CHUNK_SIZE = 25

_NSF_API_URL_TEMPLATE = (
    "https://api.nsf.gov/services/v1/awards.json?"
    "&printFields={metadata_fields}"
    "&projectOutcomesOnly={require_project_outcomes_reports}"
    "&offset={offset}"
)

###############################################################################
# LUTs

class NSFPrograms:
    Biological_Sciences = "BIO"
    Computer_and_Information_Science_and_Engineering = "CISE"
    Education_and_Human_Resources = "EHR"
    Engineering = "ENG"
    Geosciences = "GEO"
    Integrative_Activities = "OIA"
    International_Science_and_Engineering = "OISE"
    Mathematical_and_Physical_Sciences = "MPS"
    Social_Behavioral_and_Economic_Sciences = "SBE"
    Technology_Innovation_and_Partnerships = "TIP"

CFDA_NUMBER_TO_NSF_PROGRAM_NAME_LUT = {
    "47.041": NSFPrograms.Engineering,
    "47.049": NSFPrograms.Mathematical_and_Physical_Sciences,
    "47.050": NSFPrograms.Geosciences,
    "47.070": NSFPrograms.Computer_and_Information_Science_and_Engineering,
    "47.074": NSFPrograms.Biological_Sciences,
    "47.075": NSFPrograms.Social_Behavioral_and_Economic_Sciences,
    "47.076": NSFPrograms.Education_and_Human_Resources,
    "47.079": NSFPrograms.International_Science_and_Engineering,
    "47.083": NSFPrograms.Integrative_Activities,
    "47.084": NSFPrograms.Technology_Innovation_and_Partnerships,
}

NSF_PROGRAM_TO_CFDA_NUMBER_LUT = {
    code: number for number, code in CFDA_NUMBER_TO_NSF_PROGRAM_NAME_LUT.items()
}

###############################################################################

class NSF(DataSource):
    """Data source for the National Science Foundation."""    

    @staticmethod
    def _format_datetime(dt: str | datetime) -> str:
        """Parse datetime string or datetime and return NSF support datetime string."""
        return DataSource._parse_datetime(dt).strftime("%m/%d/%Y")
    
    @staticmethod
    def _format_query(
        query: str | None,
        from_datetime: str | datetime | None,
        to_datetime: str | datetime | None,
        cfda_number: str | None,
        require_project_outcomes_reports: bool,
        offset: int,
    ) -> str:
        """Format the full API string with query parameters."""
        # Fill info with always known values
        metadata_fields = ",".join(_DEFAULT_METADATA_SET)
        api_str = _NSF_API_URL_TEMPLATE.format(
            metadata_fields=metadata_fields,
            require_project_outcomes_reports=require_project_outcomes_reports,
            offset=offset,
        )

        # Handle optional parameters
        if from_datetime:
            # Parse and format
            from_dt = NSF._format_datetime(from_datetime)
            api_str += f"&dateStart={from_dt}"
        if to_datetime:
            # Parse and format
            to_dt = NSF._format_datetime(to_datetime)
            api_str += f"&dateEnd={to_dt}"
        if cfda_number:
            api_str += f"&cfdaNumber={cfda_number}"
        if query:
            api_str += f"&keyword={query}"

        return api_str
    
    @staticmethod
    def _format_dataframe(
        data: pd.DataFrame,
        query: str | None = None,
    ) -> pd.DataFrame:
        # Create column of first and last name combined
        data[DatasetFields.pi] = data["piFirstName"] + " " + data["piLastName"]

        # Drop piFirstName and piLastName columns
        data = data.drop(columns=["piFirstName", "piLastName"])

        # Format all dates as date iso format
        data["startDate"] = data["startDate"].apply(
            DataSource._format_date_for_pynder_standard
        )
        data["expDate"] = data["expDate"].apply(
            DataSource._format_date_for_pynder_standard
        )
        data["date"] = data["date"].apply(
            DataSource._format_date_for_pynder_standard,
            fmt="year",
        )

        # Add columns for query and source
        data[DatasetFields.query] = query
        data[DatasetFields.source] = "NSF"

        # Rename columns to standard
        data = data.rename(
            columns={
                "awardeeName": DatasetFields.institution,
                "date": DatasetFields.year,
                "startDate": DatasetFields.start,
                "expDate": DatasetFields.end,
                "cfdaNumber": DatasetFields.program,
                "estimatedTotalAmt": DatasetFields.amount,
                "abstractText": DatasetFields.abstract,
            }
        )

        # Create new dataframe with only the columns we want
        return data[ALL_DATASET_FIELDS]
        
    @staticmethod
    def _get_chunk(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
        cfda_number: str | None = None,
        require_project_outcomes_reports: bool = False,
        offset: int = 0,
        raise_on_error: bool = True,
    ) -> pd.DataFrame | None:
        """Get a chunk of data from the National Science Foundation."""
        # Construct the query string
        api_str = NSF._format_query(
            query,
            from_datetime,
            to_datetime,
            cfda_number,
            require_project_outcomes_reports,
            offset,
        )

        try:
            # Make the request
            resp = requests.get(api_str)

            # Get the data
            data = resp.json()["response"]
            if "award" in data:
                return_data = pd.DataFrame(data["award"])
            else:
                # If no data, return empty dataframe
                return_data = pd.DataFrame(columns=_DEFAULT_METADATA_SET)

            return NSF._format_dataframe(return_data)

        except Exception as e:
            # Handle raise on error or ignore
            if raise_on_error:
                raise e

            log.error(
                f"Error while fetching NSF data: {e}; "
                f"'raise_on_error' is False, ignoring..."
            )
        
        # Default return but make this strict
        return None

    @staticmethod
    def get_data(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
        cfda_number: str | int | None = None,
        project_outcomes_required: bool = False,
        raise_on_error: bool = True,
        tqdm_kwargs: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get data from the National Science Foundation."""
        # Continuously get chunks of data
        offset = 0
        chunks: list[pd.DataFrame] = []
        with tqdm(desc="Fetching NSF data", **(tqdm_kwargs or {})) as pbar:
            while True:
                # Get the chunk
                chunk = NSF._get_chunk(
                    query,
                    from_datetime,
                    to_datetime,
                    cfda_number,
                    project_outcomes_required,
                    offset,
                    raise_on_error,
                )
                chunks.append(chunk)

                # Break if less than chunk size
                if chunk is not None and len(chunk) < _DEFAULT_CHUNK_SIZE:
                    break

                # Update state
                offset += _DEFAULT_CHUNK_SIZE
                pbar.update(1)

        # Concatenate the chunks
        return (
            pd.concat(chunks, ignore_index=True)
            .drop_duplicates(subset="id")
            .reset_index(drop=True)
        )