#!/usr/bin/env python

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from .base import ALL_DATASET_FIELDS, DatasetFields, DataSource

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

_DEFAULT_CHUNK_SIZE = 50

# Mellon uses GraphQL for their API
# To try and debug / for documentation, use: https://graphdoc.io/
_MELLON_GRAPHQL_API_URL = "https://www.mellon.org/api/graphql"

# Bulk Query means we can get the basics of multiple grants at once
_BULK_QUERY_STATEMENT = """
query GrantFilterQuery($term: String!, $limit: Int!, $offset: Int!, $sort: SearchSort, $amountRanges: [FilterRangeInput!], $grantMakingAreas: [String!], $country: [String!], $pastProgram: Boolean, $yearRange: FilterRangeInput, $years: [Int!], $state: [String!], $ideas: [String!], $features: [String!]) {
    grantSearch(
        term: $term
        limit: $limit
        offset: $offset
        sort: $sort
        filter: {pastProgram: $pastProgram, grantMakingAreas: $grantMakingAreas, country: $country, years: $years, yearRange: $yearRange, amountRanges: $amountRanges, state: $state, ideas: $ideas, features: $features}
    ) {
        ...GrantSearchResults
    }
}

fragment GrantSearchResults on GrantSearchResultWithTotal {
    entities {
        data {
            title
            grantMakingArea
            description
            dateAwarded
            id
            grantee
        }
    }
    totalCount
}
""".strip()  # noqa: E501
_DEFAULT_BULK_PARAMS: dict = {
    "operationName": "GrantFilterQuery",
    "variables": {
        "limit": _DEFAULT_CHUNK_SIZE,
        "offset": 0,
        "term": "",
        "sort": "MOST_RELEVANT",
        "years": [],
        "grantMakingAreas": [],
        "ideas": [],
        "pastProgram": False,
        "amountRanges": [],
        "country": [],
        "state": [],
        "features": [],
    },
    "query": _BULK_QUERY_STATEMENT,
}

# Single Query means we can get the details of a single grant
# In this case, we only need the funding amount
_SINGLE_GRANT_QUERY_STATEMENT = """
query($grantId: String!) {
    grantDetails(grantId: $grantId) {
        grant {
            amount
        }
    }
}
""".strip()
_DEFAULT_SINGLE_PARAMS: dict = {
    "variables": {
        "grantId": "",
    },
    "query": _SINGLE_GRANT_QUERY_STATEMENT,
}

###############################################################################


class Mellon(DataSource):
    """Data source for the Mellon Foundation."""

    @staticmethod
    def _format_query_params(
        query: str | None,
        from_datetime: str | datetime | None,
        to_datetime: str | datetime | None,
        offset: int,
        limit: int = _DEFAULT_CHUNK_SIZE,
    ) -> dict:
        """Format the full API string with query parameters."""
        # Get years
        years = []
        if from_datetime is not None and to_datetime is None:
            to_datetime = datetime.utcnow()
        if from_datetime is not None and to_datetime is not None:
            from_datetime_parsed = DataSource._parse_datetime(from_datetime)
            to_datetime_parsed = DataSource._parse_datetime(to_datetime)

            # Get all years between from and to
            years = list(
                range(
                    from_datetime_parsed.year,
                    to_datetime_parsed.year + 1,
                )
            )

        # Copy and update params
        params = deepcopy(_DEFAULT_BULK_PARAMS)
        params["variables"]["term"] = query or ""
        params["variables"]["offset"] = offset
        params["variables"]["years"] = years
        params["variables"]["limit"] = limit

        # Return the full API string
        return params

    @staticmethod
    def _query_total_grants(
        query: str | None = None,
        from_datetime: str | datetime | None = None,
        to_datetime: str | datetime | None = None,
    ) -> int:
        try:
            # Get formatted params
            query_params = Mellon._format_query_params(
                query=query,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
                offset=0,
                limit=1,
            )

            # Request
            resp = requests.post(
                url=_MELLON_GRAPHQL_API_URL,
                json=query_params,
            )

            # Raise for status
            resp.raise_for_status()
            data = resp.json()

            # Get totalCount
            return data["data"]["grantSearch"]["totalCount"]

        except Exception as e:
            raise ValueError(f"Error while fetching total grants: {e}") from e

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame,
        query: str | None = None,
    ) -> pd.DataFrame:
        # Format all dates as date iso format
        df[DatasetFields.year] = df["dateAwarded"].apply(
            DataSource._format_date_for_pynder_standard,
            fmt="year",
        )

        # Add columns for query and source
        df[DatasetFields.query] = query
        df[DatasetFields.source] = "Mellon Foundation"

        # Rename columns to standard
        df = df.rename(
            columns={
                "grantee": DatasetFields.institution,
                "description": DatasetFields.abstract,
                "estimatedTotalAmt": DatasetFields.amount,
                "grantMakingArea": DatasetFields.program,
            }
        )

        # Make empty columns for the rest of the required fields
        for field in ALL_DATASET_FIELDS:
            if field not in df.columns:
                df[field] = None

        # Create new dataframe with only the columns we want
        return df[ALL_DATASET_FIELDS]

    @staticmethod
    def _get_funded_amount_for_grant(grant_id: str) -> float:
        # Copy and update params
        params = deepcopy(_DEFAULT_SINGLE_PARAMS)
        params["variables"]["grantId"] = grant_id

        try:
            # Make the request
            resp = requests.post(
                url=_MELLON_GRAPHQL_API_URL,
                json=params,
            )

            # Raise for status
            resp.raise_for_status()
            data = resp.json()

            # Process results
            return data["data"]["grantDetails"]["grant"]["amount"]

        except Exception as e:
            raise ValueError(f"Error while fetching grant amount: {e}") from e

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
        query_params = Mellon._format_query_params(
            query=query,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
            offset=offset,
        )

        try:
            # Make the request
            resp = requests.post(
                url=_MELLON_GRAPHQL_API_URL,
                json=query_params,
            )

            # Raise for status
            resp.raise_for_status()
            data = resp.json()

            # Process results
            chunk = pd.DataFrame(
                [item["data"] for item in data["data"]["grantSearch"]["entities"]]
            )

            # Make API calls for amount funded
            chunk["amount"] = chunk["id"].apply(Mellon._get_funded_amount_for_grant)

            return Mellon._format_dataframe(chunk)

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
        raise_on_error: bool = True,
        tqdm_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        """Get data from the National Science Foundation."""
        # First check for the total number of grants
        total_grants = Mellon._query_total_grants(
            query=query,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
        )

        # Cache dataframe results as they return
        results = []
        for offset in tqdm(
            range(0, total_grants, _DEFAULT_CHUNK_SIZE),
            **(tqdm_kwargs or {}),
        ):
            # Get the chunk
            chunk = Mellon._get_chunk(
                query=query,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
                offset=offset,
                raise_on_error=raise_on_error,
            )

            # If chunk is None, continue
            if chunk is None:
                continue

            # Append to results
            results.append(chunk)

        # Concatenate all results
        all_results = pd.concat(results, ignore_index=True)

        # Filter out any results that are not within the date range
        # if from_datetime is not None and to_datetime is not None:
        #     all_results = all_results[
        #         (all_results["dateAwarded"] >= from_datetime)
        #         & (all_results["dateAwarded"] <= to_datetime)
        #     ]

        return all_results
