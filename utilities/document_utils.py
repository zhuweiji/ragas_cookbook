from typing import List

from langchain_core.documents import Document


def filter_for_matching_header_metadata(docs: List[Document], headers: dict):
    """
    Filters a list of dictionaries based on a matching key-value pair in another dictionary.

    Args:
        data_list: A list of dictionaries with only one key-value pair each.
        match_dict: A dictionary with a single key-value pair to match against.

    Returns:
        A list of dictionaries from the data_list that have a matching key-value pair with the match_dict.
    """
    return [i for i in docs if any(
        header_type in i.metadata and
        i.metadata[header_type] == header_value
        for header_type, header_value in headers.items()
    )]
