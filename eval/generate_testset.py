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

#from qdrant_db import HybridSearcher, Qdrant
from src.utility_functions import docs_list_from_df, read_and_concatenate, update_section_with_kwargs
from src.llama_index_llm import LLMServiceManager
#from llm_answer_fixing import apply_critic_llm_validation

from pathlib import Path

current_file = Path(__file__)
repo_root = current_file.resolve().parent.parent
config_path = repo_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

ragas_models_config = config['ragas_models']

llama_index_cohere = LLMServiceManager("cohere", "command-r-plus-08-2024", "embed-english-v3.0")
llama_index_azure_openai = LLMServiceManager("azure_openai", "gpt-4o-sim", "text-embedding-ada-002")

# set up generator llm, critic llm and embeddings to create the synthetic testset.    
generator_llm = llama_index_cohere.llm
critic_llm = llama_index_azure_openai.llm
embeddings = llama_index_cohere.embed_model

def llama_index_ragas_df (input_files : list[str], text_field, metadata_fields, **kwargs) -> pd.DataFrame:
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
    return df
"""
def rag_answers_to_ragas_questions (ragas_df: pd.DataFrame, collection_name: str) -> pd.DataFrame:
    
    Parameters
    ----------
    ragas_df : Pandas df
        Conteains the synthetic testset generated through ragas, that was created using the function llama_index_ragas_df.
    collection_name : str
        The name of the qdrant collection that contains all of your data.

    Returns
    -------
    testset_df : pd.DataFrame
        the ragas_df with the answers of my RAG.

    
    # crteate testset_df
    testset = {'question': [],
               'ground_truth': [],
               'answer': [],
               'contexts': [],}
    testset_df = pd.DataFrame(testset)
    
    #Generate an answer to all of the questions from the ragas_df.
    engine = HybridSearcher()
    for index, row in ragas_df.iterrows():
        answer = engine.basic_QA_chain(collection_name ,row['question'])
        
        
        testset_row_details = {'question': row['question'],
                               'ground_truth': row['ground_truth'],
                               'answer': answer['answer'],
                               'contexts': answer['context']
            }
        
        testset_row_df = pd.DataFrame([testset_row_details])
        testset_df = pd.concat([testset_df, testset_row_df], ignore_index=True)
    
    
    return testset_df"""
       

if __name__ == '__main__':
    text_field = "paragraph_text"
    metadata_fields = ['site', 'country', 'title', 'author', 'content_publish_date']
    ragas_df = llama_index_ragas_df(['../data/espn/espn_stories.csv'], text_field, metadata_fields)
    
    
    """espn_stories_qdrant = Qdrant("espn_stories")
    espn_stories_qdrant.add_data_to_collection(['data/espn/espn_stories.csv'])
   
    questions_df = pd.read_csv('data/testsest/questions_testset.csv')
    testset = rag_answers_to_ragas_questions(questions_df, "espn_stories")
    for index, row in testset.iterrows():
        print (row['answer'])
        print ('---------------')
    
    testset.to_csv("basic_answers.csv", index=False)
    print (testset)
    
    critic_llm_answers = apply_critic_llm_validation(testset)
    critic_llm_answers.to_csv("critic_llm_answers.csv")
    print (critic_llm_answers['answer'])
    """
