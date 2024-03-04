#!/usr/bin/env python

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base import ALL_DATASET_FIELDS, DatasetFields, DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


_TEMPLETON_QUERY_API_URL = (
    "https://www.templeton.org/?"
    "limit={limit}"
)
_DEFAULT_CHUNK_SIZE = 500

_TEMPLETON_BULK_API_URL = "https://www.templeton.org/grants/grant-database"

###############################################################################


class Templeton(DataSource):
    """Data source for the Templeton Foundation."""

    @staticmethod
    def _format_query(
        query: str | None,
        limit: int = _DEFAULT_CHUNK_SIZE,
    ) -> str:
        # Fill in basic params
        query_url = _TEMPLETON_QUERY_API_URL.format(limit=limit)

        # Add query
        if query:
            query_url += f"&s={query}"

        return query_url

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame,
        query: str | None = None,
    ) -> pd.DataFrame:
        # Add columns for query and source
        df[DatasetFields.query] = query
        df[DatasetFields.source] = "Templeton"

        # Rename columns to standard
        df = df.rename(
            columns={
                "Start Year": DatasetFields.year,
                "ID": DatasetFields.id_,
                "Title": DatasetFields.title,
                "Project Leader(s)": DatasetFields.pi,
                "Grantee(s)": DatasetFields.institution,
                "Grant Amount": DatasetFields.amount,
                "Funding Area": DatasetFields.program,
            }
        )

        # Add missing columns
        for col in ALL_DATASET_FIELDS:
            if col not in df:
                df[col] = None

        # Create new dataframe with only the columns we want
        return df[ALL_DATASET_FIELDS]

    @staticmethod
    def get_data(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
        raise_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        Get data from the Templeton Foundation.

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

        Returns
        -------
        pd.DataFrame
            All grants from the Templeton Foundation for the specified time
            period and query, formatted into award_pynder standard format.
        """
        # Construct the query string
        query_url = Templeton._format_query(query=query)

        try:
            # Make the request
            resp = requests.get(query_url)

            # Convert to soup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find all a tags with rel = bookmark, keep track of the link URL
            links = soup.find_all("a", rel="bookmark")
            relevant_link_urls = [link["href"] for link in links]

            # Now query bulk API
            resp = requests.get(_TEMPLETON_BULK_API_URL)

            # Convert to soup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find the table with id "grants-table"
            table = soup.find("table", id="grants-table")

            # Pass the table into a pandas dataframe
            df = pd.read_html(StringIO(str(table)))[0]

            # Additionally attach a column for the link URL
            # Find all links by finding all "a" tags in the tbody
            tbody = table.find("tbody")
            links = tbody.find_all("a")
            all_link_urls = [link["href"] for link in links]
            df["link"] = all_link_urls

            # Subset the dataframe to only links that we have from before
            df = df[df["link"].isin(relevant_link_urls)]

            return Templeton._format_dataframe(
                df,
                query=query,
            )

        except Exception as e:
            # Handle raise on error or ignore
            if raise_on_error:
                raise e

            log.error(
                f"Error while fetching Templeton data: {e}; "
                f"'raise_on_error' is False, ignoring..."
            )

        # Filter out data that isn't in datetime range
        if from_datetime:
            # First parse the datetime
            from_dt = Templeton._parse_datetime(from_datetime)
            df = df[df[DatasetFields.year] >= from_dt.year]
        if to_datetime:
            # First parse the datetime
            to_dt = Templeton._parse_datetime(to_datetime)
            df = df[df[DatasetFields.year] <= to_dt.year]

        return df
