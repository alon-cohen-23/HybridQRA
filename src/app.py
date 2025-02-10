from flask import Flask, request, jsonify
from src.qdrant_db import HybridSearcher  # Importing your HybridSearcher class
from requests.exceptions import RequestException, ConnectionError
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.api_client import ResponseHandlingException



app = Flask(__name__)

# Initialize the HybridSearcher
searcher = HybridSearcher()

@app.route('/qa_chain', methods=['POST'])
def qa_chain():
    """
    Endpoint to call the QA_chain function.
    Expects a JSON payload with the following format:
    {
        "collection_name": "your_collection_name",
        "query": "your_query",
        "prompt": "your_prompt",
        "model": "your_model",
        "provider": "cohere" or "azure_openai"
    }
    """
    try:
        # Retrieve data from the request
        data = request.get_json()

        # Extract parameters from the request
        collection_name = data.get('collection_name')
        query = data.get('query')
        prompt = data.get('prompt')
        model = data.get('model')
        provider = data.get('provider')

        # Call QA_chain function
        kwargs = {}
        if prompt:
            kwargs['prompt'] = prompt
        if model:
            kwargs['model'] = model
        if provider:
            kwargs['provider'] = provider
        
        response = searcher.QA_chain(collection_name, query, **kwargs)        

        
        return jsonify({
            'status': 'success',
            'data': response
        })

    # Specific error for invalid Cohere LLM model
    except ValueError as e:
        if "parameter model is of type number" in str(e):
            return jsonify({
                'status': 'error',
                'message': "Cohere Error: The LLM model parameter is invalid. Please ensure it is a string. Refer to https://docs.cohere.com/reference/chat."
            }), 422
        elif "provider" in str(e):
            return jsonify({
                'status': 'error',
                'message': "Value Error: Invalid provider specified. Valid options are 'cohere' or 'azure_openai'."
            }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': f"Value Error: {str(e)}"
            }), 400


    # Qdrant connection refused
    except ResponseHandlingException as e:
        return jsonify({
            'status': 'error',
            'message': f"Qdrant Error: Unable to connect to the Qdrant server. Please ensure the Qdrant Docker container is running. Qdrant response: {str(e)}"
        }), 500

    #Qdrant collection not found
    except UnexpectedResponse as e:
        if "Collection" in str(e):
            return jsonify({
                'status': 'error',
                'message': "Qdrant Error: The specified collection does not exist. Please verify the collection name."
            }), 404
        else:
            return jsonify({
                'status': 'error',
                'message': f"Unexpected Qdrant Response: {str(e)}"
            }), 500

    # General network errors (e.g., connection issues)
    except (RequestException, ConnectionError):
        return jsonify({
            'status': 'error',
            'message': "Network Error: Unable to connect to an external service. Please check your connection and try again."
        }), 503

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Unexpected Error: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
