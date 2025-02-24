import pandas as pd
from llama_index.core import Document 
from typing import List, Dict

def create_document_from_row (row, text_field: str, metadata_fields: List[str]) -> Document:
    """

    Parameters
    ----------
    row : A row from the espn stories df .
    metadata_fields: list, the fields that will be used for the metadata of the llama_index docs.
    text_field: str, the field that will be used for the text of the llama_index docs.
    
    Returns
    -------
    doc : llama index Document, text under the text field (see defult at extract_text_from_row).
    the metadata is from the rest of the metadata_fields (see defult at extract_metadata_from_row).

    """
    text = row[text_field]
    metadata = row[metadata_fields].to_dict()
    
    doc = Document(text=text, metadata=metadata)
    return doc

def docs_list_from_df(df, text_field: str, metadata_fields: List[str], chunk_size=1000) ->List[Document]:
    "apply the create_document_from_row function to the whole df"
    
    docs_list = df.apply(
        lambda row: create_document_from_row(row, text_field, metadata_fields)
        , axis=1).to_list()
    
    return docs_list

