#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:52:55 2025

@author: aloncohen
"""

from flask import Flask, render_template, request, jsonify
from qdrant_db import Qdrant, basic_QA_chain
from qdrant_client import QdrantClient
import yaml
import os
import re

app = Flask(__name__)

# Load YAML configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)


# Configure Qdrant client
qdrant_config = config['qdrant']

client = QdrantClient(url=qdrant_config['client'])
dense_model = qdrant_config['dense_model']
sparse_model = qdrant_config['sparse_model']

@app.route('/get_collections_names', methods=['GET'])
def get_collections_names():
    try:
        collections = str(client.get_collections())
        collection_names = re.findall(r"name='([^']+)'", collections)
        return jsonify(collection_names), 200
    
    except ConnectionError as e:
        # Catch connection issues with the Qdrant service
        return jsonify({"error": str(e)}), 500    
    except Exception as e:
        # Catch all other exceptions and send a server error response
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    

@app.route('/create_collection', methods=['POST'])
def create_collection():
    try:
       
        data = request.get_json()

        # Check if the required fields are provided
        if not data.get('collection_name') or not data.get('input_files'):
            raise ValueError("Both 'collection_name' and 'input_files' must be provided and cannot be empty.")

        #Create the Qdrant collection
        collection_name = data['collection_name']
        input_files = data['input_files']
        
        # Check if the file exists 
        for path in input_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file path '{path}' is invalid or does not exist.")
        
        collection = Qdrant(collection_name)
        collection.add_data_to_collection(input_files)

        # If everything works, send success response
        response = {"message": "The collection was created successfully"}
        return jsonify(response), 200

    except KeyError as e:
        # Catch any missing fields from the incoming JSON data
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError as e:
        # Catch missing fields or invalid input in the request data
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        # Catch invalid file paths
        return jsonify({"error": str(e)}), 400
    except ConnectionError as e:
        # Catch connection issues with the Qdrant service
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Catch all other exceptions and send a server error response
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500

@app.route('/create_collection', methods=['GET'])
def rag_answer ():
    data = request.get_json()
    
    if not data.get('collection_name'):
        raise ValueError("'collection_name' must be provided and cannot be empty.")
    if not 'query' in data or not isinstance(data['query'], str) or data['query'].strip() == '':
        raise ValueError("'query' must be provided, must be a string, and cannot be an empty string.")
        
    collection_name = data['collection_name']
    query = data['query']
    
    try:
        qa_result = qa_chain.basic_QA_chain(collection_name, query)
        return jsonify(qa_result), 200

    except KeyError as e:
        # Handle missing required fields in input data
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400    
    except ConnectionError as e:
        # Handle connection errors with Qdrant or OpenAI
        return jsonify({"error": "Connection issue", "message": str(e)}), 500
    except Exception as e:
        # Catch all other errors and send a generic internal server error
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500    
        
@app.route('/compute_metrics', methods=['GET'])
def compute_ragas_metrics ():
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
        
    
