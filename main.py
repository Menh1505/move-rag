from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Hàm để tải và xử lý tài liệu
def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load()

# Hàm để chia tài liệu thành các đoạn văn
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Hàm để tạo embeddings cho văn bản
def create_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)  # Khởi tạo đối tượng embedding_model
    return embedding_model  # Trả về đối tượng embedding_model thay vì embeddings

# Hàm để tạo FAISS vector store
def create_vectorstore(chunks, embedding_model):
    # Sử dụng embedding_model (chưa gọi embed_documents) để FAISS tự động gọi embed_documents
    return FAISS.from_documents(documents=chunks, embedding=embedding_model)

# Hàm để tải mô hình GPT-2 và tokenizer
def load_gpt2_model():
    try:
        print("Loading GPT-2 model...")
        model_name = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Chỉ định pad_token là eos_token
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Hàm để sinh văn bản từ mô hình GPT-2
def generate_text(model, tokenizer, input_text, max_length=50):
    # Tạo input ids và attention mask
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).float()

    # Sinh văn bản
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length)
    
    # Trả lại văn bản đã sinh
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function to run the pipeline
def main():
    # Load and process documents
    documents = load_documents("./test.txt")
    chunks = split_documents(documents)

    # Generate embeddings and create vectorstore
    embedding_model = create_embeddings(chunks)  # Trả về đối tượng embedding_model
    vectorstore = create_vectorstore(chunks, embedding_model)  # Sử dụng embedding_model

    # Load GPT-2 model and tokenizer
    model, tokenizer = load_gpt2_model()
    if model and tokenizer:
        # Generate text from GPT-2
        input_text = "What do you know about Move code?"
        generated_text = generate_text(model, tokenizer, input_text, max_length=50)
        print("Generated Text:", generated_text)

# Run the main function
if __name__ == "__main__":
    main()
