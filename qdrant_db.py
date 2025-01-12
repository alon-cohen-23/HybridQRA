from utility_functions import create_index_dict_from_df, read_and_concatenate, convert_search_dict_to_index_dict, update_section_with_kwargs
from OpenAI_api_conn import Azure_OpenAI_api, OpenAI_api

from FlagEmbedding import FlagReranker
from typing import List, Dict
import yaml
from qdrant_client import QdrantClient
from tqdm import tqdm

#TODO : E5-large - dense model 


with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)


qdrant_config = config['qdrant']

client = QdrantClient(qdrant_config['client'])
dense_model = qdrant_config['dense_model']
sparse_model = qdrant_config['sparse_model']

openai_config = config['openai']

class Qdrant: 
    def __init__ (self, collection_name: str):
        "create a new Qdrant collection"
        self.collection_name = collection_name
        
        client.set_model(dense_model)
        client.set_sparse_model(sparse_model)
        
        # config vectors to a specific collection
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config=client.get_fastembed_vector_params(),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(), 
            on_disk_payload = True
        )
    
    def add_data_to_collection (self, input_files, chunk_size=qdrant_config['chunk_size']):
        """ add information to the collection based on the input files. 
        (see available types at the read_and_concatenate function)""" 
        
        
        #concat the input files and seperate it to the document text and metadata.
        df = read_and_concatenate(input_files)
        
        index_dict = create_index_dict_from_df(df)

        client.add(
            collection_name=self.collection_name,
            documents=index_dict['documents'],
            metadata=index_dict['metadata'],
            ids=tqdm(range(len(index_dict['documents']))),
            batch_size=chunk_size
        )  
            
            
        
reranker = FlagReranker(qdrant_config['reranker'], qdrant_config['reranker_use_fp16']) 
        
class HybridSearcher ():
    
    
    def search(self, collection_name: str, query: str, search_limit=qdrant_config['search_limit']) -> List[Dict[str, List[str]]]:
        " query the Qdrant collection and return the top answers based on the limit."
        search_result = client.query(
        collection_name=collection_name,
        query_text=query,
        query_filter=None,  
        limit=search_limit,  
    )
        
        
        retrieved_answers = [hit.metadata for hit in search_result]
        # organize retrieved context to only two keys: document and metadata.
        retrieved_answers = [convert_search_dict_to_index_dict(item) for item in retrieved_answers]
        
        return retrieved_answers    
    
        
    def search_with_rerank(self, collection_name: str, query: str, reranker_limit = qdrant_config['reranker_limit']) -> List[str]:
        """
        Parameters
        ----------
        query: the query that has been asked in the serach function.
        
        Returns
        -------
        rellevant_contexts: List
        the top 3 paragraphs sorted in a descending oreder based
        on the score of the bge-reranker-v2-m3 reranking model.
        """
        
        raw_contexts = self.search(collection_name,query)
        
        # Compute scores for each context
        scores = [reranker.compute_score([query, str(item)], normalize=True) for item in raw_contexts]
        
        scored_texts = list(zip(scores, raw_contexts))
        
        #Sort the list of tuples by score in descending order
        sorted_scored_texts = sorted(scored_texts, key=lambda x: x[0], reverse=True)
        
        sorted_texts = [text for score, text in sorted_scored_texts]
        rellevant_contexts = sorted_texts[:reranker_limit]
        
        return rellevant_contexts
      
    
    def basic_QA_chain (self, collection_name: str, query: str, **kwargs) -> Dict[str, str]:
        """
        Parameters
        ----------
        collection_name : str
            the name of the rellevant Qdrant collection.
        query : str
            the question you want to ask.
        **kwargs: dict, available keys:
            - basic_instructions: basic instructions to help the llm to provide a quality answer.
            - llm: the llm that will be used to generate the answer.
            - conn: The type of openAI api you use, can be only 'Azure_OpenAI' or 'OpenAI'.
        
        Returns
        -------
        qa_dict : Dict
            contains 3 keys: query - the same query from the input.
            context - the context that helped the llm to answer the query.
            answer - the answer that the llm generated.

        """
        # Access openai configuration from YAML
        openai_config = config['openai']
        updated_config = update_section_with_kwargs(openai_config, **kwargs)
        
        basic_instructions = updated_config['basic_instructions']
        llm = updated_config['llm']
        
        
        contexts = self.search_with_rerank(collection_name, query)
        contexts = str(contexts)
    
        messages = [{"role": "system", "content": basic_instructions},
                   {"role": "user", "content": "Question: " + query},
                   {"role": "user", "content": "Contexts: " +contexts}]
        
        api_conn = updated_config['conn']
        if api_conn == 'Azure_OpenAI': 
            answer = Azure_OpenAI_api(messages, llm)
        elif api_conn == 'OpenAI':
            answer = OpenAI_api(messages, llm)
        else:
            raise ValueError ("Your api conn must be 'OpenAI' or 'Azure_OpenAI' depend on your key, please change it in the settings or through config.yaml.")
        
        qa_dict = {'question': query, 'context': contexts, 'answer': answer}
        
        return qa_dict
        

        
if __name__ =='__main__':

   q = Qdrant("alon")
   q.add_data_to_collection(["data/espn/sample_espn.csv"])
  
   client = QdrantClient(url="http://localhost:6333")
   
   
    
    
    
    
   
