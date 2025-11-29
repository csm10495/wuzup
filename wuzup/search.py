"""
Home to search-related functionality.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from dateutil.parser import parse
from ddgs.ddgs import DDGS
from ddgs.exceptions import DDGSException

from wuzup.ai import AISupport
from wuzup.exceptions import MissingRequiredKeyError

log = logging.getLogger(__name__)


@dataclass(eq=False)
class SearchResult(AISupport):
    """
    A wrapper around an individual search result.
    """

    title: str
    url: str
    body: str | None = None
    date: datetime | None = None
    source: str | None = None
    image: str | None = None
    thumbnail: str | None = None

    @classmethod
    def from_ddgs_result(cls, ddgs_result: dict[str, str]) -> "SearchResult":
        """
        Creates a SearchResult from a DDGS result dictionary.
        Raises MissingRequiredKeyError if required keys are missing.
        """
        title = ddgs_result.get("title")
        if not title:
            raise MissingRequiredKeyError(f"DDGS result did not have a valid title: {ddgs_result}")

        url = ddgs_result.get("href") or ddgs_result.get("url") or ddgs_result.get("content")
        if not url:
            raise MissingRequiredKeyError(f"DDGS result did not have a valid url: {ddgs_result}")

        date = ddgs_result.get("date") or ddgs_result.get("published")
        if date:
            date = parse(date)

        # Not all searches have all fields, so we allow some to be empty.
        return cls(
            title=title,
            url=url,
            date=date,
            body=ddgs_result.get("body") or ddgs_result.get("description") or ddgs_result.get("info"),
            source=ddgs_result.get("source")
            or ddgs_result.get("uploader")
            or ddgs_result.get("author")
            or ddgs_result.get("publisher"),
            image=ddgs_result.get("image"),
            thumbnail=ddgs_result.get("thumbnail"),
        )

    def _coerce_for_ai_equivalence(self):
        data = super()._coerce_for_ai_equivalence()
        if "date" in data:
            del data["date"]
        return data


class Searcher:
    """
    A wrapper around search functionality for a query.
    """

    def __init__(self, query: str, typ: str, search_args: tuple = (), search_kwargs: dict[str, str] = {}) -> None:
        """
        Initializer for Searcher that takes in a query and a type of search. Type can be 'text', 'news', etc.
        We also have search_args and search_kwargs to pass to the underlying DDGS search method.
        """
        self.query = query
        self.typ = typ

        self._search_method = getattr(DDGS, typ, None)
        if self._search_method is None:
            raise ValueError(f"Search type '{typ}' is not supported. Try using 'text' or 'news' instead")

        self.search_args = search_args
        self.search_kwargs = search_kwargs
        self._ddgs = DDGS()

        log.debug(f"Searcher with query='{query}', type='{typ}' initialized.")

    def search(self) -> list[SearchResult]:
        """
        Performs the search and returns a list of SearchResult objects.
        """
        try:
            results = self._search_method(self._ddgs, query=self.query, *self.search_args, **self.search_kwargs)
        except DDGSException:
            log.debug("Search failed with DDGSException", exc_info=True)
            results = []

        ret_list = []

        for r in results:
            try:
                ret_list.append(SearchResult.from_ddgs_result(r))
            except MissingRequiredKeyError as e:
                log.debug(f"Skipping search result due to missing required key: {e}")

        return ret_list
