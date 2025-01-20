#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:51:53 2024

@author: aloncohen
"""

from pathlib import Path
import yaml
from qdrant_db import Qdrant
from qdrant_client import QdrantClient
from utility_functions import update_section_with_kwargs
from generate_testset import llama_index_ragas_df, rag_answers_to_ragas_questions
from rag_evaluation import df_evaluation_by_chunk
from llm_answer_fixing import apply_critic_llm_validation


current_file = Path(__file__)
repo_root = current_file.resolve().parent.parent
config_path = repo_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


# Configure Qdrant client
qdrant_config = config['qdrant']

client = QdrantClient(url=qdrant_config['client'])
dense_model = qdrant_config['dense_model']
sparse_model = qdrant_config['sparse_model']


main_config = config['main']


def main(**kwargs):
    """
    main function that runs through all of the RAG pipeline.
    it indexes the input data given from the input files, 
    generate the testset and evaluate the results using ragas metrics.

    Parameters
    ----------
    **kwargs : dict, available keys:
        - collection_name: str, the name of the qdrant collection.
        - input_files: list, the dataframe files that will be converted to llama index docs (csv, xlsx...)..
        - metadata_fields: list, the fields that will be used for the metadata of the llama_index docs.
        - text_field: str, the field that will be used for the text of the llama_index docs.
        - test_size: int, how many questions you want to produce.
        - distributions: dict, got three fields: simple, reasoning, multi_context.
          the sum of their values sould be equals to 1 with each value represent the part of each question type.
         - metrics: list, the ragas metrics that will be used to evaluate the RAG pipeline. 
         - chunks_amount: int, the amount of chuncks the dataframe will be splitted to during the evaluation, exeptions from the API.
         - use_critic_llm: bool, if sets to True it double check testset_df['answer'] using the function apply_critic_llm_validation.  
    Returns
    -------
    dataframe file that contains the following columns:
        question, answer, ground_truth, contexts (that were retrieved to the llm) and the rellevant metrics.

    """
    # update main_config based on the given kwargs
    updated_config = update_section_with_kwargs(main_config, **kwargs)
    
    # create the qdrant collection and index the data
    qdrant_collection = Qdrant(updated_config['collection_name'])
    qdrant_collection.add_data_to_collection(updated_config['input_files'])
    
    # generate the synthetic testset
    testset_config = {'test_size': updated_config['test_size'], 'distributions': updated_config['distributions']}
    ragas_synthetic_questions_df = llama_index_ragas_df(updated_config['input_files'], **testset_config)
    testset_df = rag_answers_to_ragas_questions(ragas_synthetic_questions_df, updated_config['collection_name'])
    
    if updated_config['use_critic_llm']:
        testset_df = apply_critic_llm_validation(testset_df)
    
    # evaluate the testset_df 
    eval_df = df_evaluation_by_chunk(testset_df, updated_config['metrics'], chunks_amount=1)
    
    return eval_df
    
if __name__ =='__main__':
    eval_df = main()
    
    
    
   



    
    
    
   


