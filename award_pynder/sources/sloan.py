#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base import ALL_DATASET_FIELDS, DatasetFields, DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

_SLOAN_API_URL = (
    "https://sloan.org/grants-database?"
    "dynamic=1"
    "&order_by=approved_at"
    "&order_by_direction=desc"
    "&limit={limit}"
    "&page={page}"
)
_DEFAULT_CHUNK_SIZE = 3000

###############################################################################


class Sloan(DataSource):
    """Data source for the Sloan Foundation."""

    @staticmethod
    def _format_query(
        query: str | None,
        offset: int,
        limit: int = _DEFAULT_CHUNK_SIZE,
    ) -> str:
        # Fill in basic params
        query_url = _SLOAN_API_URL.format(limit=limit, page=offset)

        # Add query
        if query:
            query_url += f"&keywords={query}"

        return query_url

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame,
        query: str | None = None,
    ) -> pd.DataFrame:
        # Add columns for query and source
        df[DatasetFields.query] = query
        df[DatasetFields.source] = "Sloan"

        # Add missing columns
        for col in ALL_DATASET_FIELDS:
            if col not in df:
                df[col] = None

        # Create new dataframe with only the columns we want
        return df[ALL_DATASET_FIELDS]

    @staticmethod
    def _query_total_grants(
        query: str | None,
    ) -> int:
        try:
            # Construct params for a single query
            query_url = Sloan._format_query(
                query=query,
                offset=1,
                limit=1,
            )

            # Make the request
            resp = requests.get(query_url)

            # Parse HTML for td with "results-count" class
            soup = BeautifulSoup(resp.text, "html.parser")
            return int(
                soup.find(
                    "td",
                    class_="results-count",
                )
                .text.replace(",", "")
                .replace("Grants", "")
                .strip()
            )

        except Exception as e:
            raise ValueError(f"Error while fetching total grants: {e}") from e

    @staticmethod
    def _get_chunk(
        query: str | None = None,
        offset: int = 1,
        raise_on_error: bool = True,
    ) -> pd.DataFrame | None:
        # Construct the query string
        query_url = Sloan._format_query(
            query=query,
            offset=offset,
        )

        try:
            # Make the request
            resp = requests.get(query_url)

            # Convert to soup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find the data list container
            data_list = soup.find("ul", class_="data-list")
            awards = data_list.find_all("li", recursive=False)

            # Iter over each li to collect rows
            rows = []
            for li in awards:
                # Get the header info
                header = li.find("header")

                # Collect the grantee information (stored in div.grantee)
                grantee = header.find("div", class_="grantee").text
                grantee = grantee.replace("grantee: ", "").strip()

                # Collect the amount information (stored in div.amount)
                amount = header.find("div", class_="amount").text
                amount = float(
                    amount.replace("amount: ", "")
                    .replace("$", "")
                    .replace(",", "")
                    .strip()
                )

                # Collect the year information (stored in div.year)
                year = header.find("div", class_="year").text
                year = int(year.replace("year: ", "").strip())

                # Get the details info
                details = li.find("div", class_="details")

                # Collect the description (stored in the div "brief-description")
                description = details.find(
                    "div", class_="brief-description"
                ).text.strip()

                # Collect the id (stored in the div attribute
                # "data-accordian-group" for div with class "details")
                id_ = details["data-accordian-group"].replace("grant-", "").strip()

                # Find the grid
                grid = details.find("div", class_="grid")

                # Collect the program (stored in div.grid > first ul > first li)
                program = grid.find("ul").find("li").text
                program = program.replace("Program", "").strip()

                # Take all of the text from the second ul
                sub_program_and_pi = grid.find_all("ul")[1].text.strip()

                # Find the index of the word "Investigator"
                pi_index = sub_program_and_pi.find("Investigator")

                # Only keep text after the word "Investigator"
                pi = sub_program_and_pi[pi_index + len("Investigator") :].strip()

                # Add row
                rows.append(
                    {
                        DatasetFields.institution: grantee,
                        DatasetFields.amount: amount,
                        DatasetFields.year: year,
                        DatasetFields.title: description,
                        DatasetFields.id_: int(id_),
                        DatasetFields.program: program,
                        DatasetFields.pi: pi,
                    }
                )

            # Sleep for a second
            time.sleep(2)

            return Sloan._format_dataframe(
                pd.DataFrame(rows),
                query=query,
            )

        except Exception as e:
            # Handle raise on error or ignore
            if raise_on_error:
                raise e

            log.error(
                f"Error while fetching Sloan data: {e}; "
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
        Get data from the Sloan Foundation.

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
            All grants from the Sloan Foundation for the specified time
            period and query, formatted into award_pynder standard format.
        """
        # Continuously get chunks of data
        offset = 0
        chunks: list[pd.DataFrame] = []

        # Get total
        total = Sloan._query_total_grants(
            query=query,
        )

        # Iter chunks
        for offset in tqdm(
            range(0, total, _DEFAULT_CHUNK_SIZE),
            **(tqdm_kwargs or {}),
        ):
            # Get the chunk
            chunk = Sloan._get_chunk(
                query=query,
                offset=offset,
                raise_on_error=raise_on_error,
            )

            # If chunk is None, continue
            if chunk is None:
                continue

            chunks.append(chunk)

        # Concatenate the chunks
        if len(chunks) == 0:
            return pd.DataFrame(columns=ALL_DATASET_FIELDS)

        # Concat and filter out years not in range
        df = pd.concat(chunks, ignore_index=True).reset_index(drop=True)
        if from_datetime:
            # First parse the datetime
            from_dt = Sloan._parse_datetime(from_datetime)
            df = df[df[DatasetFields.year] >= from_dt.year]
        if to_datetime:
            # First parse the datetime
            to_dt = Sloan._parse_datetime(to_datetime)
            df = df[df[DatasetFields.year] <= to_dt.year]

        return df
