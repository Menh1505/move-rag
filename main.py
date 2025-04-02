from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from markdown_it import MarkdownIt
from dotenv import dotenv_values
from langchain.schema import Document
import os
import pickle

ENV = dotenv_values(".env")
print("Environment Variables Loaded:", ENV)
# Thiết lập API key cho Anthropic
os.environ["ANTHROPIC_API_KEY"] = ENV['ANTHROPIC_API_KEY']

# Đường dẫn lưu trữ FAISS vectorstore trong thư mục làm việc
FAISS_PATH = os.path.join(os.getcwd(), "faiss_index")

# Hàm tải và đọc file Markdown từ thư mục
def load_markdown_files(directory_path):
    md_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                md_files.append(file.read())
    return md_files

# Hàm phân tích cú pháp Markdown
def parse_markdown(md_text):
    md = MarkdownIt()
    tokens = md.parse(md_text)
    elements = {"headings": [], "paragraphs": [], "code_blocks": []}
    
    current_heading = None
    for token in tokens:
        if token.type == 'heading_open':
            current_heading = token.tag  # Lưu cấp độ tiêu đề (h1, h2, h3, ...)
        elif token.type == 'inline' and current_heading:
            elements["headings"].append({"level": current_heading, "content": token.content})
            current_heading = None
        elif token.type == 'paragraph_open':
            elements["paragraphs"].append(token.content)
        elif token.type == 'fence':
            elements["code_blocks"].append({"language": token.info, "content": token.content})
    return elements

# Hàm xử lý Markdown và chuyển đổi thành văn bản
def process_markdown_files(md_files):
    processed_documents = []
    for md_text in md_files:
        elements = parse_markdown(md_text)
        # Kết hợp các phần tử Markdown thành một chuỗi văn bản
        text = ""
        for heading in elements["headings"]:
            text += f"{'#' * int(heading['level'][1])} {heading['content']}\n"
        for paragraph in elements["paragraphs"]:
            text += f"{paragraph}\n"
        for code_block in elements["code_blocks"]:
            text += f"```{code_block['language']}\n{code_block['content']}\n```\n"
        
        # Tạo đối tượng Document
        document = Document(page_content=text, metadata={"source": "markdown"})
        processed_documents.append(document)
    return processed_documents

# Hàm chia nhỏ tài liệu thành các đoạn văn
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Hàm tạo hoặc tải FAISS vectorstore
def create_or_load_faiss_vectorstore(documents=None, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Kiểm tra nếu FAISS đã tồn tại
    if os.path.exists(f"{FAISS_PATH}.pkl"):
        print("Loading existing FAISS index...")
        with open(f"{FAISS_PATH}.pkl", "rb") as f:
            faiss_index = pickle.load(f)
        return faiss_index
    else:
        print("Creating new FAISS index...")
        if not documents or len(documents) == 0:
            raise ValueError("No documents found to create FAISS index. Please check your input data.")
        
        # Tạo embeddings
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        faiss_index = FAISS.from_documents(documents=documents, embedding=embedding_model)
        
        # Lưu FAISS index
        with open(f"{FAISS_PATH}.pkl", "wb") as f:
            pickle.dump(faiss_index, f)
        return faiss_index

# Hàm truy xuất dữ liệu từ FAISS
def retrieve_relevant_chunks(query, faiss_index, top_k=5):
    results = faiss_index.similarity_search(query, k=top_k)
    return "\n".join([result.page_content for result in results])

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
        # Tải và xử lý file Markdown
        md_files = load_markdown_files(directory_path)
        processed_texts = process_markdown_files(md_files)
        
        # Tạo hoặc tải FAISS vectorstore
        faiss_index = create_or_load_faiss_vectorstore(processed_texts)
        
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