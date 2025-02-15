import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import hashlib
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "last_request" not in st.session_state:
    st.session_state.last_request = 0

def get_pdf_text(pdf_docs):
    """Extract text from PDF files with enhanced error handling"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"Empty page found in {pdf.name}")
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            raise
    return text if text else None

def get_text_chunks(text):
    """Split text into optimized chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_documents(pdf_docs, existing_index=False):
    """Process PDF documents with enhanced validation and error recovery"""
    index_path = f"faiss_index_{st.session_state.session_id}"
    
    try:
        with st.spinner("Analyzing documents..."):
            # Validate input
            if not pdf_docs:
                raise ValueError("No documents provided")
                
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text:
                raise ValueError("No text extracted from PDFs")

            text_chunks = get_text_chunks(raw_text)
            if not text_chunks:
                raise ValueError("Failed to split text into chunks")

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_chunks = []

            # Deduplication check
            for chunk in text_chunks:
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                if chunk_hash not in st.session_state.processed_hashes:
                    new_chunks.append(chunk)
                    st.session_state.processed_hashes.add(chunk_hash)

            if not new_chunks:
                st.info("No new content found in uploaded documents")
                return False

            # Handle index operations
            if existing_index:
                if not os.path.exists(index_path):
                    raise FileNotFoundError("No existing index to update")
                vector_store = FAISS.load_local(index_path, embeddings)
                vector_store.add_texts(new_chunks)
                st.success("Updated existing knowledge base")
            else:
                vector_store = FAISS.from_texts(new_chunks, embedding=embeddings)
                st.success("Created new knowledge base")

            # Atomic commit
            temp_store = vector_store
            temp_store.save_local(index_path)
            st.session_state.vector_store = temp_store
            return True

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        if 'temp_store' in locals():
            del temp_store
        return False

def get_conversational_chain():
    """Create the QA chain with corrected safety controls"""
    prompt_template =  """
You are an enthusiastic academic assistant helping students prioritize study topics and resources. 
Follow these guidelines:


1. Response Style:
- Start with encouraging phrases: " Sure!", "Let's optimize your study plan!", "Great  question! 📚"

- Maintain supportive yet professional tone
- Use occasional educational emojis (📖, ✨, 🎯)


2. Topic Analysis:
When asked about important topics:
a) Analyze document structure: headings, sections, repetitions
b) Identify 3-5 key topics with:
   - Frequency in document
   - Detailed explanations/examples
   - Summary sections
c) Highlight foundational vs advanced concepts


3. Study Recommendations:
For each key topic:

- Suggest study approaches: 
  "Focus on understanding X through..." 
  "Practice Y-type problems for..."

- Warn about common pitfalls: 
  "Students often struggle with Z because..."


4. Resource Guidance:
For reference materials:

- First check document citations/references
- If none found, suggest:
  "Standard references for this subject include..."
  "Recommended resources from academic sources:"

- Always add disclaimer: "Verify syllabus requirements"


5. Content Rules:
- Start with document-based priorities

- For unknown topics: "While not emphasized here, typically important..."
- When unsure: "Let me check the materials... [analyzes context]"

- Clear limitations: "The documents don't specify resources, but..."

6. Format Requirements:
- Organize in numbered lists

- Use bold indicators without markdown: *Important* 
- Separate sections with line breaks

Example Response Structure:
"Great question! Let's break this down:

🎯 Key Topics (from your materials):

1. Topic A: Appears 15x, detailed examples in Chapter 3
2. Topic B: Core concept covered in 5 sections

3. Topic C: Summary section emphasizes this

📚 Study Priorities:

- Master Topic B first (foundational)
- Practice Topic A through [specific method from doc]

- Review Topic C's diagrams regularly

📖 Recommended Resources:

1. *Essential Text*: [Book mentioned in doc citations]
2. *Supplementary*: [Standard field textbook]

3. *Online*: [Domain-specific learning site]

✨ Pro Tip: Focus on..."

Context: {context}

Question: {question}

Answer:
"""
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.6,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        }
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def sanitize_input(text):
    """Basic input sanitization"""
    return text.replace('<', '&lt;').replace('>', '&gt;').strip()

def handle_query(user_question):
    """Process user question with rate limiting and enhanced validation"""
    # Rate limiting
    if time.time() - st.session_state.last_request < 5:
        st.warning("Please wait 5 seconds between requests")
        return "Request throttled - please wait"
    
    try:
        st.session_state.last_request = time.time()
        sanitized_question = sanitize_input(user_question)
        
        if not st.session_state.vector_store:
            raise ValueError("Knowledge base not initialized")

        docs = st.session_state.vector_store.similarity_search(sanitized_question, k=3)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": sanitized_question},
            return_only_outputs=True
        )
        
        return response["output_text"].strip()
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "An error occurred while processing your request"

def main():
    """Main application layout with enhanced UI"""
    st.set_page_config(
        page_title="PDF Insight AI Pro",
        page_icon="📘",
        layout="centered"
    )
    
    st.title("📘 PDF Insight AI Pro")
    st.caption("Enterprise Document Intelligence powered by Gemini")
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                st.caption(message["metadata"])

    # Sidebar controls
    with st.sidebar:
        st.header("Document Management")
        pdf_docs = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Max 100 pages per document, 10 documents max"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_new = st.button("Create New KB", help="Start fresh knowledge base")
        with col2:
            process_update = st.button("Update Existing", help="Add to current knowledge base")
        
        if process_new or process_update:
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    success = process_documents(pdf_docs, existing_index=process_update)
            else:
                st.warning("Please upload documents first")
        
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.chat_history.clear()
            st.experimental_rerun()

    # Handle user input
    if user_question := st.chat_input("Ask about your documents"):
        if not st.session_state.vector_store:
            st.warning("Please process documents first")
            return
        
        # Add user question to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        try:
            start_time = time.time()
            response = handle_query(user_question)
            processing_time = time.time() - start_time
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "metadata": f"Processed in {processing_time:.2f}s | Session ID: {st.session_state.session_id}"
            })
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Response generated in {processing_time:.2f} seconds")

        except Exception as e:
            st.error(f"Query failed: {str(e)}")

if __name__ == "__main__":
    main()



    
