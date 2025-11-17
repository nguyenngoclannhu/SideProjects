from urllib.parse import urlencode
import requests

class CustomModel:
    def __init__(self):
        self.inference_url = "http://localhost:8000/chat"
        self.rag_pipelien = "http://localhost:8000/search-documents"

    def encoded_query(self, query):
        return query.replace(" ", "%20").replace("'", "%27")
    
    def search_documents(self, query):
        querystring = {
                        "question": query,
                        "num_results": 5,
                        "temperature": 0.7
                    }

        headers = {"authorization": "Basic YWRtaW46YWRtaW4="}
        response = requests.post(self.rag_pipelien, headers=headers, json=querystring)
        print(response.json())
        return response.json()

    def chat(self, query):
        # encoded_query = urlencode(query)
        querystring = {
                        "question": query,
                        "num_results": 1,
                        "temperature": 0.7
                    }

        headers = {"authorization": "Basic YWRtaW46YWRtaW4="}
        response = requests.post(self.inference_url, headers=headers, json=querystring)
        print(response.json())
        return response.json()
    