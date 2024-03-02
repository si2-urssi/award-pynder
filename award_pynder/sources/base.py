"""Data sources module for the award_pynder package."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import pandas as pd
from dateutil.parser import parse as dateutil_parse

###############################################################################


class DatasetFields:
    institution = "institution"
    pi = "pi"
    year = "year"
    start = "start"
    end = "end"
    program = "program"
    amount = "amount"
    id_ = "id"
    title = "title"
    abstract = "abstract"
    query = "query"
    source = "source"


ALL_DATASET_FIELDS = [
    field_value
    for field_name, field_value in vars(DatasetFields).items()
    if not field_name.startswith("_")
]

###############################################################################


class DataSource(ABC):
    """Abstract base class for data sources."""

    @staticmethod
    def _parse_datetime(dt: str | datetime) -> datetime:
        if isinstance(dt, str):
            return dateutil_parse(dt)

        return dt

    @staticmethod
    def _format_date_for_pynder_standard(
        dt: str | datetime,
        fmt: Literal["year", "date"] = "date",
    ) -> str | int | None:
        if dt is None:
            return None
        if isinstance(dt, str):
            dt = dateutil_parse(dt)

        if fmt == "year":
            return dt.year

        return dt.date().isoformat()

    @staticmethod
    @abstractmethod
    def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format the data to standard across all funders."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_data() -> pd.DataFrame:
        """Get data from the source."""
        raise NotImplementedError()
