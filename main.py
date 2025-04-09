from langchain_anthropic import ChatAnthropic
from dotenv import dotenv_values
from markdown_process import process_and_create_faiss, retrieve_relevant_chunks
import os

# Tải biến môi trường
ENV = dotenv_values(".env")
print("Environment Variables Loaded:", ENV)
os.environ["ANTHROPIC_API_KEY"] = ENV['ANTHROPIC_API_KEY']

# Hàm sinh văn bản với Claude AI
def generate_text_with_anthropic(input_text, max_tokens=100):
    try:
        chat_model = ChatAnthropic(model="claude-2")  # Sử dụng mô hình Claude-2
        response = chat_model.invoke(input_text, max_tokens=max_tokens)
        return response
    except Exception as e:
        print(f"Error generating text with Anthropic: {e}")
        return None

# Hàm chính
def main():
    # Đường dẫn thư mục chứa tài liệu Markdown
    directory_path = "./documents"

    try:
        # Gọi hàm xử lý Markdown và tạo FAISS index
        faiss_index = process_and_create_faiss(directory_path)
        
        # Nhập câu hỏi từ người dùng
        query = input("Enter your question about Move: ")
        
        # Truy xuất dữ liệu liên quan từ FAISS
        relevant_chunks = retrieve_relevant_chunks(query, faiss_index)
        print("\nRelevant Chunks Retrieved:")
        print(relevant_chunks)
        
        # Gửi câu hỏi và dữ liệu liên quan đến Claude AI
        input_text = f"Here is some context about Move:\n{relevant_chunks}\n\nQuestion: {query}"
        generated_text = generate_text_with_anthropic(input_text, max_tokens=300)
        
        # Hiển thị kết quả
        if generated_text:
            print("\nGenerated Response:")
            print(generated_text)
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Chạy chương trình
if __name__ == "__main__":
    main()
