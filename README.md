# MoveLazy AI

MoveLazy AI is a Python-based project that processes Markdown documents, creates vector embeddings using FAISS, and generates responses to user queries using Anthropic's Claude AI. This tool is designed to help users interact with and retrieve insights from Markdown-based documentation.

## Features

- **Markdown Parsing**: Reads and processes Markdown files to extract headings, paragraphs, and code blocks.
- **FAISS Vectorstore**: Creates or loads a FAISS index for efficient similarity searches.
- **Text Generation**: Uses Anthropic's Claude AI to generate responses based on user queries and relevant document chunks.
- **Customizable**: Supports configurable chunk sizes, overlap, and embedding models.

## Project Structure
├── .env # Environment variables file 
├── .gitignore # Git ignore file 
├── faiss_index.pkl # Serialized FAISS index (generated after first run) 
├── main.py # Main script 
├── documents/ # Directory containing Markdown documents 
│ ├── ...

## Requirements

- Python 3.8+
- Required Python packages (see [Installation](#installation))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movelazy-ai.git
   cd movelazy-ai
   ```

2. Install dependencies:
`pip install -r requirements.txt`

3. Create a .env file in the root directory and add your Anthropic API key:
`ANTHROPIC_API_KEY=your_anthropic_api_key`

4. Place your Markdown files in the documents/ directory.

## Usage
Run the main script to start the application:

`python main.py` or `python3 main.py`

## Workflow
1. The script loads Markdown files from the documents/ directory.
2. It processes the Markdown files into structured text.
3. A FAISS vectorstore is created or loaded to enable similarity searches.
4. Users can input a query, and the script retrieves relevant chunks from the documents.
5. The query and relevant chunks are sent to Anthropic's Claude AI for text generation.

## Example
```bash
Enter your question about Move: What is the purpose of the Move language?
```
Output:
```
Relevant Chunks Retrieved:
# Introduction
Move is a programming language designed for ...

Generated Response:
Move is a language optimized for secure and efficient execution of smart contracts...
```

## Configuration
- **Chunk Size and Overlap:** Modify the `split_documents` function in `main.py` to adjust `chunk_size` and `chunk_overlap`.
- **Embedding Model:** Change the `embedding_model_name` parameter in the `create_or_load_faiss_vectorstore` function to use a different HuggingFace model.

## Dependencies
- [langchain](https://github.com/langchain-ai/langchain)
- [markdown-it-py](https://github.com/executablebooks/markdown-it-py)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Anthropic-API](https://www.anthropic.com/)

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [langchain](https://github.com/langchain-ai/langchain) for document processing and vectorstore integration.
- [Anthropic](https://www.anthropic.com/) for their Claude AI API.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.

## Contact
For questions or support, please open an issue or contact [dinhthienmenh1505@gmail.com].