import os
import google.generativeai as genai
import gradio as gr
# from pypdf import PdfReader
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



gemini_key= os.getenv("GEMINI_API_KEY")
print(gemini_key)

genai.configure(api_key=gemini_key)

# Define System Prompt for Astrology
system_prompt = """You are an expert astrologer with deep knowledge of astrology, planetary positions, and horoscope reading. 
You provide accurate astrological insights, guidance on zodiac signs, and help people understand their birth charts."""

# Initialize Gemini Model with the System Prompt
model = genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=system_prompt)


def pdf_reader(base_path):
    documents = []
    if os.path.exists(base_path) and os.path.isdir(base_path):  # Ensure it's a valid directory
        loader = DirectoryLoader(
            base_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = os.path.basename(base_path)  # Use 'astro_content' as doc_type
            documents.append(doc)
    else:
        print(f"Warning: Folder '{base_path}' does not exist or is not a directory.")

    return documents
     
    
base_path = "C:/Users/raclo/astro_ai/astro_content"
documents=pdf_reader(base_path)

# for doc in documents:
#     print(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=gemini_key)
vectorstore = Chroma.from_documents(chunks, embeddings)

# create a new Chat with OpenAI
llm = GoogleGenerativeAI(temperature=0.7, model='gemini-2.0-flash')

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat).launch()
# view = gr.ChatInterface(fn=run_conversation,type='messages').launch()

