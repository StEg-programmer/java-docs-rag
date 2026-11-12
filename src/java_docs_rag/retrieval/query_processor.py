"""Developer A: query preparation before embedding."""


class QueryProcessor:
    def process(self, query: str) -> str:
        """Strip boundary whitespace, preserving the query's case and internal text.

        Raise TypeError for non-strings and ValueError for empty or blank strings.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        stripped = query.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped
