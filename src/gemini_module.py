import json
import requests
import os
import faiss
import numpy as np
from typing import List, Dict

class GeminiProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def generate_text(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Gửi prompt đến Google Gemini API và nhận phản hồi.
        """
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "key": self.api_key
        }
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generation_config": {
                "max_output_tokens": max_tokens
            }
        }

        try:
            response = requests.post(self.endpoint, headers=headers, params=params, json=data)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            result = response.json()
            # Trích xuất văn bản từ phản hồi
            return result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return None


class DataProcessor:
    def __init__(self, blocks_path: str, metadata_path: str, vector_path: str):
        self.blocks_path = blocks_path
        self.metadata_path = metadata_path
        self.vector_path = vector_path
        self.index = None

    def load_data(self) -> tuple:
        """
        Tải dữ liệu từ các file JSON và FAISS.
        """
        try:
            # Đọc blocks và metadata
            with open(self.blocks_path, 'r') as f:
                blocks = json.load(f)
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            # Tải FAISS index
            self.index = faiss.read_index(self.vector_path)

            return blocks, metadata
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Tìm kiếm các blocks tương tự trong FAISS index.
        """
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")
        
        D, I = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        blocks, metadata = self.load_data()
        for idx in I[0]:
            results.append({
                'content': blocks[idx],
                'metadata': metadata[idx]
            })
        return results
