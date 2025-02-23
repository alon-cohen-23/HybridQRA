#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:23:19 2024

@author: aloncohen
"""

import pandas as pd
from llama_index.core import Document 
from typing import List, Dict
import time


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

def create_index_dict_from_df (docs_df: pd.DataFrame(), text_field: str, metadata_fields: List[str]) -> Dict[str, List[str]]:
    """
    
    Parameters
    ----------
    docs_df : df that contains the text and the metadata columns.
    metadata_fields: list, the fields that will be used for the metadata of the qdrant docs.
    text_field: str, the field that will be used for the text of the qdrant docs.
    
    Returns
    -------
    index_dict : Dict
        contains two keys: documents: list of the text of the documents
                            metadata: list of the metadata dictionaries that matches each document.
    """
    
    documents = docs_df[text_field].to_list()
    metadata = docs_df[metadata_fields].apply(lambda row: row.to_dict(), axis=1).tolist()
    
    index_dict = {'documents': documents, 'metadata':metadata}
    
    return index_dict

def convert_search_dict_to_index_dict (search_dict: dict) -> Dict[str, List[str]]:
    """convert the serach dict that created from quering the qdrant db
    to a dict with the same format as the index dit from create_index_dict_from_df"""
    
    index_dict = {
        "document": search_dict["document"],
        "metadata": {k: v for k, v in search_dict.items() if k != "document"}
    }
    return index_dict

def dict_to_document_str (doc_dic: dict):
    """
    Parameters
    ----------
    doc_dic : dictionary that contains all of the documnt metadata and the document itself
    as it's keys from the create_index_dict_from_df function.

    Returns
    -------
    doc_str : string based on this template:
        document: {document content}
        the document metadata is:
        {metadata}    .

    """
    # Initialize an empty string
    result_str = ""

    result_str += f"document: {doc_dic['document']}" + '\n'
    del doc_dic['document']

    result_str += "the document metadata is:" + '\n'

    #Convert all other keys and values to strings and concatenate
    result_str += ", ".join([f"'{str(key)}': '{value}'" for key, value in doc_dic.items()])   
    result_str = result_str.strip(", ")    
    
    return result_str


def read_and_concatenate(file_paths: list) -> pd.DataFrame():
    """
    concatenate a list of dataframe files in different formats (csv, xlsx...)
    
    Parameters
    ----------
    file_paths : List of paths to dataframe files with the same fields.

    Raises
    ------
    ValueError if the columns are not identical.

    Returns
    -------
    concatenated_df.

    """
    # Define a dictionary to map file extensions to their respective read functions
    read_functions = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.html': lambda file_path: pd.read_html(file_path)[0], 
        '.parquet': pd.read_parquet
    }

    dataframes = []

    # Loop through the list of file paths
    for file_path in file_paths:
        # Find the appropriate read function based on the file extension
        for ext, read_func in read_functions.items():
            if file_path.endswith(ext):
                df = read_func(file_path)
                dataframes.append(df)
                break
        else:
            raise ValueError(f"Unsupported file extension for file: {file_path}")

    # Check if all DataFrames have the same columns
    first_df_columns = dataframes[0].columns
    for df in dataframes:
        if not first_df_columns.equals(df.columns):
            raise ValueError("Not all DataFrames have the same columns")

   
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    return concatenated_df
    
def update_section_with_kwargs(section_config: dict, **kwargs) -> dict:
    """
    Updates a specific section of the configuration with values from kwargs.

    Parameters
    ----------
    section_config : dict
        The original configuration section to be updated.
    **kwargs : dict
        The keyword arguments that may override the section configuration.

    Returns
    -------
    updated_section : dict
        The updated section configuration dictionary.
    """
    updated_section = section_config.copy()  # Start with a copy of the section configuration
    
    for key, value in kwargs.items():
        if key in updated_section:
            if isinstance(updated_section[key], dict) and isinstance(value, dict):
                # If both section config and kwargs value are dicts, recursively update
                updated_section[key] = update_section_with_kwargs(updated_section[key], **value)
            else:
                updated_section[key] = value  # Override the section config value with kwargs value
        else:
            pass

    return updated_section
    

if __name__ == "__main__":
    df = read_and_concatenate(['../data/espn/espn_stories.csv'])
    text_field = "paragraph_text"
    metadata_fields = ['site', 'country', 'title', 'author', 'content_publish_date']
    
    docs_list = docs_list_from_df(df, text_field, metadata_fields)
    print (len(docs_list))    
