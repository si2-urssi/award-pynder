#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from copy import deepcopy
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from .base import ALL_DATASET_FIELDS, DatasetFields, DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

_NIH_API_URL = "https://api.reporter.nih.gov/v2/projects/Search"
_DEFAULT_METADATA_SET = [
    "Organization",
    "ProjectNum",
    "ProjectSerialNum",
    "FiscalYear",
    "ProjectStartDate",
    "ProjectEndDate",
    "ProjectTitle",
    "AgencyCode",
    "AbstractText",
    "ContactPiName",
    "AwardAmount",
    "AwardNoticeDate",
]
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_PARAMS: dict = {
    "criteria": {
        "award": {
            "award_notice_date": {
                "from_date": None,
                "to_date": None,
            }
        },
        "exclude_subprojects": True,
        "advanced_text_search": {
            "search_text": None,
            "operator": "advanced",
            "search_field": "abstracttext",
        },
    },
    "include_fields": _DEFAULT_METADATA_SET,
    "limit": _DEFAULT_CHUNK_SIZE,
    "offset": 0,
}

###############################################################################


class NIH(DataSource):
    """Data source for the National Institute of Health."""

    @staticmethod
    def _format_datetime(dt: str | datetime) -> str:
        """Parse datetime string or datetime and return NIH support datetime string."""
        return DataSource._parse_datetime(dt).date().isoformat()

    @staticmethod
    def _format_query(
        query: str | None,
        from_datetime: str | datetime | None,
        to_datetime: str | datetime | None,
        offset: int,
        limit: int = _DEFAULT_CHUNK_SIZE,
    ) -> dict:
        """Format the full API string with query parameters."""
        # Fill info with always known values
        params = deepcopy(_DEFAULT_PARAMS)
        params["criteria"]["award"]["award_notice_date"]["from_date"] = (
            NIH._format_datetime(from_datetime) if from_datetime else None
        )
        params["criteria"]["award"]["award_notice_date"]["to_date"] = (
            NIH._format_datetime(to_datetime) if to_datetime else None
        )
        params["criteria"]["advanced_text_search"]["search_text"] = query or ""
        params["limit"] = limit
        params["offset"] = offset

        return params

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame,
        query: str | None = None,
    ) -> pd.DataFrame:
        # Format all dates as date iso format
        df[DatasetFields.start] = df["project_start_date"].apply(
            DataSource._format_date_for_pynder_standard,
        )
        df[DatasetFields.end] = df["project_end_date"].apply(
            DataSource._format_date_for_pynder_standard,
        )

        # Add columns for query and source
        df[DatasetFields.query] = query
        df[DatasetFields.source] = "NIH"

        # Rename columns to standard
        df = df.rename(
            columns={
                "organization": DatasetFields.institution,
                "contact_pi_name": DatasetFields.pi,
                "fiscal_year": DatasetFields.year,
                "agency_code": DatasetFields.program,
                "award_amount": DatasetFields.amount,
                "project_num": DatasetFields.id_,
                "project_title": DatasetFields.title,
                "abstract_text": DatasetFields.abstract,
            }
        )

        # Cast amount to float
        df[DatasetFields.amount] = df[DatasetFields.amount].astype(float)

        # Create new dataframe with only the columns we want
        return df[ALL_DATASET_FIELDS]

    @staticmethod
    def _query_total_grants(
        query: str | None,
        from_datetime: str | datetime | None,
        to_datetime: str | datetime | None,
    ) -> int:
        """Query the total number of grants from the National Science Foundation."""
        try:
            # Construct params for a single query
            params = NIH._format_query(
                query=query,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
                offset=0,
                limit=1,
            )

            # Make the request
            resp = requests.post(_NIH_API_URL, json=params)

            # Get the data
            data = resp.json()

            return data["meta"]["total"]

        except Exception as e:
            raise ValueError(f"Error while fetching total grants: {e}") from e

    @staticmethod
    def _get_chunk(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
        offset: int = 0,
        raise_on_error: bool = True,
    ) -> pd.DataFrame | None:
        """Get a chunk of data from the National Science Foundation."""
        # Construct the query string
        params = NIH._format_query(
            query=query,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
            offset=offset,
        )

        try:
            # Make the request
            resp = requests.post(_NIH_API_URL, json=params)

            # Get the data
            data = resp.json()["results"]

            # For each result, extract the org_name and change nulls to None
            rows = []
            for result in data:
                result["organization"] = result["organization"]["org_name"]
                rows.append(result)

            if len(rows) > 0:
                return_data = pd.DataFrame(rows)
            else:
                # If no data, return empty dataframe
                return_data = pd.DataFrame(columns=_DEFAULT_METADATA_SET)

            # Sleep for a second
            time.sleep(2)

            return NIH._format_dataframe(return_data, query=query)

        except Exception as e:
            # Handle raise on error or ignore
            if raise_on_error:
                raise e

            log.error(
                f"Error while fetching NIH data: {e}; "
                f"'raise_on_error' is False, ignoring..."
            )

        # Default return but make this strict
        return None

    @staticmethod
    def get_data(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
        raise_on_error: bool = True,
        tqdm_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        """
        Get data from the National Institute of Health.

        Parameters
        ----------
        query : str, optional
            The query string to search for.
        from_datetime : str or datetime, optional
            The start date for the search.
        to_datetime : str or datetime, optional
            The end date for the search.
        raise_on_error : bool, optional
            Whether to raise an error if the request fails.
        tqdm_kwargs : dict, optional
            Keyword arguments to pass to tqdm.

        Returns
        -------
        pd.DataFrame
            All grants from the National Institute of Health for the specified time
            period and query, formatted into award_pynder standard format.
        """
        # Continuously get chunks of data
        offset = 0
        chunks: list[pd.DataFrame] = []

        # Get total
        total = NIH._query_total_grants(
            query=query,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
        )

        # Handle too many
        if total >= 10000:
            raise ValueError(
                f"Total grants is {total}, which is too many to fetch. "
                "Please narrow your search."
            )

        # Iter chunks
        for offset in tqdm(
            range(0, total, _DEFAULT_CHUNK_SIZE),
            **(tqdm_kwargs or {}),
        ):
            # Get the chunk
            chunk = NIH._get_chunk(
                query=query,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
                offset=offset,
                raise_on_error=raise_on_error,
            )

            # If chunk is None, continue
            if chunk is None:
                continue

            chunks.append(chunk)

        # Concatenate the chunks
        return pd.concat(chunks, ignore_index=True).reset_index(drop=True)
