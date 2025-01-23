from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_entity_recall, context_precision, context_recall, context_relevancy
from ragas import evaluate

from datasets import Dataset

import pandas as pd
import numpy as np
import time
import yaml
from pathlib import Path
from src.llm_providers.llama_index_llm import LLMServiceManager



current_file = Path(__file__)
repo_root = current_file.resolve().parent.parent
config_path = repo_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

ragas_models_config = config['ragas']

llama_index_azure_openai = LLMServiceManager("azure_openai", ragas_models_config['eval_llm'],
                                             ragas_models_config['eval_embeddings'])

# set up generator llm, critic llm and embeddings to create the synthetic testset.    
llm = llama_index_azure_openai.llm
embeddings = llama_index_azure_openai.embed_model



def assemble_ragas_dataset(testset_df: pd.DataFrame) -> Dataset:
    "convert the testset_df generated from generate_testset.rag_answers_to_ragas_questions to HF dataset"
    
    question_list = testset_df['question'].to_list()
    truth_list = testset_df['ground_truth'].to_list()
    context_list = testset_df['contexts'].to_list()
    context_list = [[context] for context in context_list]
    rag_answer_list = testset_df['answer'].to_list()


    # Create a HuggingFace Dataset from the ground truth lists.
    ragas_ds = Dataset.from_dict({"question": question_list,
                            "answer": rag_answer_list,
                            "contexts": context_list,
                            "ground_truth": truth_list
                            })
    return ragas_ds


def df_evaluation(testset_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """

    Parameters
    ----------
    testset_df : pd.DataFrame
        the testset_df generated from generate_testset.rag_answers_to_ragas_questions.
    metrics : list
        A list of ragas metrics (can only be taken from the metrics imported above.

    Returns
    -------
    df_score : pd.DataFrame
        contains the testset_df with the score of all of the metrics to each row.

    """
    # set the llm and the embeder 
    ragas_llm = LlamaIndexLLMWrapper(llm)
    ragas_emb = LlamaIndexEmbeddingsWrapper(embeddings=embeddings)

    str_metrics = [func.name for func in metrics]

    # Change the default models used for each metric.
    for metric in str_metrics:
        globals()[metric].llm = ragas_llm
        globals()[metric].embeddings = ragas_emb

    dataset = assemble_ragas_dataset(testset_df)
    
    score = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings)
    df_score = score.to_pandas()
    
    return df_score


metrics_dict = {
        'faithfulness': faithfulness,
        'answer_correctness': answer_correctness,
        'answer_relevancy': answer_relevancy,
        'context_entity_recall': context_entity_recall,
        'context_precision': context_precision,
        'context_recall': context_recall,
        'context_relevancy': context_relevancy
    }

def df_evaluation_by_chunk (testset_df:pd.DataFrame ,metrics: list[str], chunks_amount=5) -> pd.DataFrame:
    """
    evaluate every chunck seperately by calling df_evaluation to not overload Azure OpenAI API.
    very important! len(testset_df) has to be divisible by the chunks_amount."""
    
    
    metrics = [metrics_dict[metric] for metric in metrics if metric in metrics_dict]
    testset_df = testset_df.fillna('There is no answer.')
    
    # Split the DataFrame into smaller DataFrames, to avoid exeptions from the API
    dfs = np.array_split(testset_df, chunks_amount)
    
    eval_dfs =[]
    for df_chunk in dfs:
        part_eval = df_evaluation(df_chunk, metrics)
        eval_dfs.append(part_eval)
        time.sleep(10)
    
    concatenated_eval_df = pd.concat(eval_dfs, ignore_index=True)
    
    return concatenated_eval_df
    
   
    
if __name__ =='__main__':

    metrics = [
               'answer_correctness',
               'answer_relevancy',
               'context_precision',
               'context_recall',
               'context_relevancy']
    
    
    df = pd.read_csv('../data/testsest/command-r-plus-08-2024_answers2.csv')
   
    eval_df = df_evaluation_by_chunk(df, metrics)
    eval_df.to_csv('command-r-plus-08-2024_answers_results2.csv', index=False)
    
    print (eval_df[metrics].mean())
    
    
        
   

    
        
    
