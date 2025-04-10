from dotenv import dotenv_values
from src.gemini_module import GeminiProcessor, DataProcessor
import numpy as np
import logging

# Cấu hình logging để ghi vào file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Ghi log vào file app.log
        # Không sử dụng StreamHandler để không in ra console
    ]
)

# Tải biến môi trường
ENV = dotenv_values(".env")

# Hàm chính để gọi model Gemini
def main():
    # Đường dẫn đến các tệp dữ liệu
    BLOCKS_PATH = "./data/blocks.json"
    METADATA_PATH = "./data/metadata.json"
    VECTOR_PATH = "./data/vectors.faiss"
    API_KEY = ENV['GOOGLE_API_KEY']

    # Khởi tạo các module
    gemini = GeminiProcessor(api_key=API_KEY)
    data_processor = DataProcessor(BLOCKS_PATH, METADATA_PATH, VECTOR_PATH)

    try:
        # Tải dữ liệu và FAISS index
        blocks, metadata = data_processor.load_data()

        # Kiểm tra xem blocks và metadata có được tải thành công không
        if blocks is None or metadata is None:
            logging.error("Failed to load blocks or metadata.")
            return

        # Nhập câu hỏi từ người dùng
        query = input("Enter your question: ")

        # Chuyển câu hỏi thành vector (giả sử bạn có mô hình embedding để tạo vector)
        query_vector = np.random.rand(384).astype('float32')  # Thay thế bằng vector thực tế

        # Tìm kiếm các đoạn văn bản liên quan
        logging.info("Searching for similar blocks...")
        try:
            similar_blocks = data_processor.search_similar(query_vector)
            logging.info("Successfully retrieved %d similar blocks.", len(similar_blocks))
        except Exception as e:
            logging.error("Error during searching similar blocks: %s", e)
            return
        
        # Kiểm tra xem có blocks nào được tìm thấy không
        if not similar_blocks:
            logging.warning("No similar blocks found.")
            print("No similar blocks found.")
            return

        # In nội dung của similar_blocks để kiểm tra
        logging.debug("Similar blocks content: %s", similar_blocks)

        # Kiểm tra cấu trúc của similar_blocks
        for block in similar_blocks:
            if not isinstance(block, dict):
                logging.error("Expected a dictionary but got: %s", block)
            if 'content' not in block:
                logging.error("Block does not contain 'content': %s", block)

        # Tạo context từ nội dung của các blocks
        try:
            context = "\n".join([block['content']['content'] for block in similar_blocks if isinstance(block['content'], dict) and 'content' in block['content']])
            logging.debug("Generated context from similar blocks: %s", context)
        except Exception as e:
            logging.error("Error creating context from similar blocks: %s", e)
            return

        # Kiểm tra nếu context rỗng
        if not context.strip():
            logging.warning("Context is empty. Skipping request to Gemini API.")
            print("No valid content found in similar blocks.")
            return

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
