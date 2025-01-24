#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 22:00:43 2024

@author: aloncohen
"""

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

import pandas as pd
import yaml

from src.utils.utility_functions import docs_list_from_df, read_and_concatenate, update_section_with_kwargs
from src.utils.logger import get_logger

from src.llm_providers.llama_index_llm import LLMServiceManager
from src.qdrant_db import HybridSearcher, QdrantCollectionManager

from pathlib import Path

logger = get_logger()

current_file = Path(__file__)
repo_root = current_file.resolve().parent.parent
config_path = repo_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

ragas_models_config = config['ragas']

llama_index_cohere = LLMServiceManager("cohere", ragas_models_config['generator_llm'],
                    ragas_models_config['generator_embeddings'])
llama_index_azure_openai = LLMServiceManager("azure_openai", ragas_models_config['critic_llm'],
                    ragas_models_config['eval_embeddings'])

# set up generator llm, critic llm and embeddings to create the synthetic testset.    
generator_llm = llama_index_cohere.llm
critic_llm = llama_index_azure_openai.llm
embeddings = llama_index_cohere.embed_model

def create_synthetic_ragas_df (input_files : list[str], text_field, metadata_fields, **kwargs) -> pd.DataFrame:
    """
    generates a synthetic testset using ragas API based on the data in the input files
    Parameters
    ----------
    input_files : list of files that will be converted to llama index docs (csv, xlsx...).
    
    **kwargs: dict, available keys:
        - test_size: int, how many questions you want to produce.
        - distributions: dict, got three fields: simple, reasoning, multi_context.
          the sum of their values sould be equals to 1 with each value represent the part of each question type.

    Returns
    -------
    df : pd.DataFrame
        Conteains the synthetic testset generated through ragas.

    """
    # set testset_config and update it according to the kwargs.
    testset_config = config['testset']
    testset_config = update_section_with_kwargs(testset_config, **kwargs)
    
    # set up the llama_index docs that the synthetic testset will be built on.      
    df = read_and_concatenate(input_files)
    docs = docs_list_from_df(df, text_field, metadata_fields)
    
    generator = TestsetGenerator.from_llama_index(
        generator_llm,
        critic_llm,
        embeddings
    )
    logger.info(f"Started to Generate synthetic ragas df based on f{input_files}.")
    
    # create the testset
    testset = generator.generate_with_llamaindex_docs(
        docs,
        test_size=testset_config['test_size'],
        raise_exceptions=False,
        with_debugging_logs=False,
        distributions={simple: testset_config['distributions']['simple'],
                       reasoning: testset_config['distributions']['reasoning'],
                       multi_context: testset_config['distributions']['multi_context']},
    )
    
    df = testset.to_pandas()
    logger.info("Generated synthetic ragas df successfully.")
    
    return df

def rag_answers_to_ragas_questions (ragas_df: pd.DataFrame, collection_name: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    ragas_df : Pandas df
        Conteains the synthetic testset generated through ragas, that was created using the function create_synthetic_ragas_df.
    collection_name : str
        The name of the qdrant collection that contains all of your data.

    Returns
    -------
    testset_df : pd.DataFrame
        the ragas_df with the answers of my RAG.
    """
    
    # crteate testset_df
    testset = {'question': [],
               'ground_truth': [],
               'answer': [],
               'contexts': [],}
    testset_df = pd.DataFrame(testset)
    
    logger.info(f"Answering synthetic ragas questions for collection: {collection_name}")
    #Generate an answer to all of the questions from the ragas_df.
    engine = HybridSearcher()
    for index, row in ragas_df.iterrows():
        answer = engine.QA_chain(collection_name, row['question'])
        
        
        testset_row_details = {'question': row['question'],
                               'ground_truth': row['ground_truth'],
                               'answer': answer['answer'],
                               'contexts': answer['context']
            }
        
        testset_row_df = pd.DataFrame([testset_row_details])
        testset_df = pd.concat([testset_df, testset_row_df], ignore_index=True)
    
    logger.info("Finished answering synthetic ragas questions.")
    
    return testset_df
       

if __name__ == '__main__':
    
    text_field = "paragraph_text"
    metadata_fields = ['title', 'content_publish_date']
    input_files = ['../data/espn/espn_stories.csv']
    
    qdrant = QdrantCollectionManager()
    qdrant.create_collection("espn_stories")
    qdrant.add_data_to_collection("espn_stories", input_files,
                                  text_field, metadata_fields)
    
    print ("collection was loaded")
    ragas_df_path = "../data/testsest/testset_questions.csv"
    
    ragas_df = pd.read_csv(ragas_df_path)
    
    testset_df = rag_answers_to_ragas_questions (ragas_df, "espn_stories")
    testset_df.to_csv("../data/testsest/command-r-plus-08-2024_answers2.csv", index=False)
    
    
    
  
