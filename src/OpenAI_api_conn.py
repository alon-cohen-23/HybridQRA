#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:56:19 2024

@author: aloncohen
"""

import os
from llama_index.llms.azure_openai import AzureOpenAI as llama_index_Azure  
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

def llama_index_llm_connection (azure_deployment_model: str) -> llama_index_Azure:
    "connect to llama_index AzureOpenAI, imported as llama_index_Azure"
        
    llm = llama_index_Azure(
        azure_deployment = azure_deployment_model,
        api_key = os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version = os.environ['AZURE_OPENAI_API_VERSION'],
       )

    return llm

def llama_index_embedding_Connection(azure_deployment_embedding: str) -> AzureOpenAIEmbedding:
    "connect to llama_index AzureOpenAIEmbedding"
    
    azure_embed_model = AzureOpenAIEmbedding(
        azure_deployment=azure_deployment_embedding, 
        api_key = os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version = os.environ['AZURE_OPENAI_API_VERSION'],
    )

    return azure_embed_model
    


from openai import AzureOpenAI

def Azure_OpenAI_api (messages: list, azure_deployment_model: str) -> str:
    "connect to openai api and return the llm answer to the prompt (messages)"
    client = AzureOpenAI(
        azure_deployment = azure_deployment_model,
        api_key = os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version = os.environ['AZURE_OPENAI_API_VERSION'],
       )

    completion = client.chat.completions.create(
        model = azure_deployment_model,  # e.g. gpt-35-instant
        messages= messages
        )    
    return completion.choices[0].message.content

import openai

def OpenAI_api (messages: list, openai_deployment_model:str):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai_deployment_model = openai_deployment_model
    
    response = openai.ChatCompletion.create(
    model=openai_deployment_model, 
    messages=messages
    )

    return (response.choices[0].message['content'])

import cohere

COHERE_API_KEY = os.environ['COHERE_API_KEY']
co = cohere.ClientV2(api_key=COHERE_API_KEY)


def cohere_api (documents:list, model="command-r-plus-08-2024" ):

    # Add the user message
    basic_instructions = "please answer the given question using only the given contexts."
    message = "Where do the tallest penguins live?"
    messages = [{"role": "system", "content": basic_instructions},
                {"role": "system", "content": str(documents)},
                {"role": "user", "content": message}]
    response = co.chat(
        model=model,
        messages=messages,
    )
    #return (response.message.content[0].text)
    return response

if __name__ == "__main__":

    # Retrieve the documents
    documents = [
        {
            "data": {
                "title": "Tall penguins",
                "snippet": "Emperor penguins are the tallest.",
            }
        },
        {
            "data": {
                "title": "Penguin habitats",
                "snippet": "Emperor penguins only live in Antarctica.",
            }
        },
        {
            "data": {
                "title": "What are animals?",
                "snippet": "Animals are different from plants.",
            }
        },
    ]
    
    res = cohere_api(documents)
    print (res.message)
    
    
