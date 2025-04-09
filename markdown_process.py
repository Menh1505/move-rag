from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from markdown_it import MarkdownIt
from langchain.schema import Document
import os
import pickle

# Đường dẫn lưu trữ FAISS vectorstore
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

# Hàm xử lý toàn bộ Markdown và tạo FAISS index
def process_and_create_faiss(directory_path):
    # Tải và xử lý file Markdown
    md_files = load_markdown_files(directory_path)
    processed_texts = process_markdown_files(md_files)
    
    # Tạo hoặc tải FAISS vectorstore
    faiss_index = create_or_load_faiss_vectorstore(processed_texts)
    return faiss_index
