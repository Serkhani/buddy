from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        transformed_tries: number of times the query has been transformed
    """

    question: str
    generation: str
    documents: List[str]
    transformed_tries: int