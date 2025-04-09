from dotenv import dotenv_values
from src.gemini_module import GeminiProcessor, DataProcessor
import numpy as np
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải biến môi trường
ENV = dotenv_values(".env")
logging.info("Environment Variables Loaded: %s", ENV)

# Hàm chính để gọi model Gemini
def main():
    # Đường dẫn đến các tệp dữ liệu
    BLOCKS_PATH = "./data/blocks.json"
    METADATA_PATH = "./data/metadata.json"
    VECTOR_PATH = "./data/vectors.faiss"
    API_KEY = ENV['ANTHROPIC_API_KEY']

    # Khởi tạo các module
    logging.info("Initializing GeminiProcessor and DataProcessor...")
    gemini = GeminiProcessor(api_key=API_KEY)
    data_processor = DataProcessor(BLOCKS_PATH, METADATA_PATH, VECTOR_PATH)

    try:
        # Tải dữ liệu và FAISS index
        logging.info("Loading data from %s, %s, and %s...", BLOCKS_PATH, METADATA_PATH, VECTOR_PATH)
        blocks, metadata = data_processor.load_data()
        logging.debug("Loaded %d blocks and %d metadata entries.", len(blocks), len(metadata))

        # Nhập câu hỏi từ người dùng
        query = input("Enter your question: ")
        logging.info("User query: %s", query)

        # Chuyển câu hỏi thành vector (giả sử bạn có mô hình embedding để tạo vector)
        # Ở đây, bạn cần thay thế `query_vector` bằng vector thực tế từ mô hình embedding
        query_vector = np.random.rand(768).astype('float32')  # Thay thế bằng vector thực tế
        logging.debug("Generated query vector: %s", query_vector)

        # Tìm kiếm các đoạn văn bản liên quan
        logging.info("Searching for similar blocks...")
        similar_blocks = data_processor.search_similar(query_vector)
        logging.debug("Found %d similar blocks.", len(similar_blocks))

        context = "\n".join([block['content'] for block in similar_blocks])
        logging.debug("Context for Gemini API: %s", context)

        # Gửi câu hỏi và ngữ cảnh đến Gemini API
        prompt = f"Context: {context}\n\nQuestion: {query}"
        logging.info("Sending prompt to Gemini API...")
        response = gemini.generate_text(prompt)

        # Hiển thị kết quả
        if response:
            logging.info("Received response from Gemini API.")
            print("\nGemini Response:")
            print(response)
        else:
            logging.warning("No response from Gemini API.")
            print("No response from Gemini API.")
    except Exception as e:
        logging.error("An error occurred: %s", e)

# Chạy chương trình
if __name__ == "__main__":
    main()
