import os
import time
import asyncio
import nest_asyncio
import PyPDF2
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Tuple, Optional, Any
from io import BytesIO
from qdrant_client.http import models
import uuid
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging
import tempfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# Add MongoDB connection
from pymongo import MongoClient
import hashlib
import secrets
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable nested event loops
nest_asyncio.apply()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "")
try:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["pdf_database"]  # Use the correct database name
    products_collection = db["pdf_documents"]  # Use the correct collection name
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
    products_collection = None

class URLProcessor:
    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def extract_content(url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                element.decompose()
            
            content_tags = soup.find_all(['article', 'main', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            processed_content = []
            for tag in content_tags:
                text = tag.get_text(strip=True, separator=' ')
                if text and len(text.split()) > 5:
                    processed_content.append(text)
            
            return '\n\n'.join(processed_content)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return ""

class DocumentProcessor:
    @staticmethod
    def process_pdf(file_content: bytes) -> str:
        try:
            logger.info("Starting PDF processing")
            content = []
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content.append(page_text)
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1}: {str(page_error)}")
                    continue
            
            final_content = "\n\n".join(content)
            logger.info(f"Successfully processed PDF with {len(pdf_reader.pages)} pages")
            return final_content
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return ""

class ProductContext:
    def __init__(self):
        self.products = []
        self.general_content = ""

    def update_content(self, products, general_content):
        """Update content while maintaining unique products"""
        # Add new products while avoiding duplicates
        existing_names = {p['name'] for p in self.products}
        for product in products:
            if product['name'] not in existing_names:
                self.products.append(product)
                existing_names.add(product['name'])
        
        # Append new content with a separator if there's existing content
        if self.general_content and general_content:
            self.general_content += "\n\n=== New Content ===\n\n" + general_content
        else:
            self.general_content = general_content

    def get_combined_context(self):
        if not self.products:  # If no products, return just the general content
            return self.general_content.strip()
            
        context = "Available Products:\n"
        for i, product in enumerate(self.products, 1):
            context += f"\n{i}. Product Name: {product['name']}"
            context += f"\n   Price: {product['price']}"
            if product.get('category'):
                context += f"\n   Category: {product['category']}"
            if product.get('description'):
                context += f"\n   Description: {product['description']}"
            if product.get('image'):
                context += f"\n   Image URL: {product['image']}"
            if product.get('productUrl'):
                context += f"\n   Product URL: {product['productUrl']}"
            if product.get('_id'):
                context += f"\n   _id: {product['_id']}"
            context += "\n"
        
        return context.strip()

# Global product context
product_context = ProductContext()

def initialize_vertex_ai():
    try:
        creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        if creds_json:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                json.dump(json.loads(creds_json), temp_file)
                temp_creds_path = temp_file.name
            
            credentials = service_account.Credentials.from_service_account_file(temp_creds_path)
            os.unlink(temp_creds_path)
            
        else:
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not creds_path:
                raise FileNotFoundError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
                
            credentials = service_account.Credentials.from_service_account_file(creds_path)
        
        aiplatform.init(
            credentials=credentials,
            project=" ",
            location=""
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {str(e)}")

async def fetch_products_from_mongodb():
    try:
        logger.info("Fetching products from MongoDB")
        if products_collection is None:
            logger.error("MongoDB connection not available")
            return [], "MongoDB connection not available"
        
        products = list(products_collection.find({}))
        
        # Convert MongoDB documents to product format
        formatted_products = []
        for product in products:
            # Handle PDF documents structure
            formatted_product = {
                "_id": str(product["_id"]),
                "name": product.get("filename", ""),
                "price": "N/A",  # PDF documents don't have prices
                "image": "https://s7d9.scene7.com/is/image/dollargeneral/dg-placeholder-150x209",
                "productUrl": "#",
                "category": "Document",
                "description": product.get("text_content", "")[:200] + "..." if product.get("text_content") else ""
            }
            formatted_products.append(formatted_product)
        
        logger.info(f"Successfully fetched {len(formatted_products)} products from MongoDB")

        # Update the product context
        product_context.products = formatted_products
        product_context.general_content = "Products loaded from database."
        
        return formatted_products, product_context.get_combined_context()
    except Exception as e:
        logger.error(f"Error fetching products from MongoDB: {e}", exc_info=True)
        return [], "Error fetching products from database"

def process_content(content: str) -> Tuple[List[str], List[str]]:
    try:
        logger.info(f"Processing content length: {len(content)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)
        
        # Ensure all chunks are strings
        chunks = [str(chunk) for chunk in chunks]
        
        return chunks, chunks  # Return both the chunks and a sample
        
    except Exception as e:
        logger.error(f"Error in process_content: {e}", exc_info=True)
        raise

def initialize_vector_store(text_chunks: List[str], collection_name: str) -> Optional[Qdrant]:
    try:
        initialize_vertex_ai()
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        
        # Initialize Qdrant client with cloud configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Delete collection if it exists
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created new collection: {collection_name}")
        
        # Convert text chunks to strings if they aren't already
        text_chunks = [str(chunk) if not isinstance(chunk, str) else chunk for chunk in text_chunks]
        
        # Create vector store
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        # Add texts in smaller batches
        batch_size = 20  # Reduced batch size for better stability
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            try:
                vector_store.add_texts(
                    texts=batch,
                    ids=[str(uuid.uuid4()) for _ in batch]
                )
                logger.info(f"Processed batch {i//batch_size + 1} of {(len(text_chunks)-1)//batch_size + 1}")
            except Exception as batch_error:
                logger.error(f"Error processing batch starting at index {i}: {batch_error}")
                continue
        
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}", exc_info=True)
        return None

class HybridRetriever:
    def __init__(self, vector_store, texts: List[str], k: int = 4):
        self.bm25_retriever = BM25Retriever.from_texts(texts)
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.3, 0.7]
        )
    
    async def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            documents = await self.ensemble_retriever.aget_relevant_documents(query)
            return documents
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return await self.vector_retriever.aget_relevant_documents(query)

# Flask app setup
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.secret_key = os.getenv("SECRET_KEY", "1234")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

def init_messages():
    if 'messages' not in session:
        session['messages'] = []
        session['messages'].append({
            'sender': 'bot',
            'text': "Hello, I'm your Compliance Assistant. Let's Explore our Products and Solutions!",
            'timestamp': datetime.now().strftime("%I:%M %p")
        })

@app.route("/", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for('prompt'))

    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        if email == "user@example.com" and password == "1234":
            session['logged_in'] = True
            session['user'] = email.split('@')[0]
            session.permanent = True
            
            # Automatically load products from MongoDB after successful login
            try:
                # Run the fetch_products_from_mongodb function using asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    products, content = loop.run_until_complete(fetch_products_from_mongodb())
                finally:
                    loop.close()
                
                if products:
                    # Process content for vector search
                    text_chunks, chunks_sample = process_content(content)
                    if not text_chunks:
                        return render_template("login.html", error="Failed to process product data")
                    
                    collection_name = f"chatbot_{uuid.uuid4().hex[:8]}"
                    vector_store = initialize_vector_store(text_chunks, collection_name)
                    
                    if not vector_store:
                        return render_template("login.html", error="Failed to initialize vector store")
                    
                    session['collection_name'] = collection_name
                    session['content_processed'] = True
                    session['text_chunks_sample'] = chunks_sample[:5] if chunks_sample else []
                    session['mongodb_products'] = True
                    session.modified = True
                    
                    return redirect(url_for('prompt'))
                else:
                    # If no products found, still proceed to prompt page
                    session['no_products'] = True
                    return redirect(url_for('prompt'))
            except Exception as e:
                logger.error(f"Error auto-loading products: {e}")
                # Even if there's an error, proceed to prompt page
                return redirect(url_for('prompt'))
        else:
            error = "Invalid email or password"
            return render_template("login.html", error=error)
        
    return render_template("login.html")

@app.route("/prompt", methods=["GET", "POST"])
def prompt():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    init_messages()

    # Add warning message if no products were loaded
    if session.pop('no_products', False):
        session['messages'].append({
            'sender': 'bot',
            'timestamp': datetime.now().strftime("%I:%M %p")
        })
        session.modified = True

    if request.method == "POST":
        user_query = request.form.get('prompt_input')
        
        if user_query:
            try:
                session['messages'].append({
                    'sender': 'user',
                    'text': user_query,
                    'timestamp': datetime.now().strftime("%I:%M %p")
                })

                collection_name = session.get('collection_name')
                text_chunks_sample = session.get('text_chunks_sample', [])
                
                if not collection_name or not text_chunks_sample:
                    response = "Please wait while I load product information from the database."
                    
                    # Try to load products again if they weren't loaded initially
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        products, content = loop.run_until_complete(fetch_products_from_mongodb())
                        
                        if products:
                            text_chunks, chunks_sample = process_content(content)
                            new_collection_name = f"chatbot_{uuid.uuid4().hex[:8]}"
                            vector_store = initialize_vector_store(text_chunks, new_collection_name)
                            
                            session['collection_name'] = new_collection_name
                            session['content_processed'] = True
                            session['text_chunks_sample'] = chunks_sample[:5]
                            session['mongodb_products'] = True
                            session.modified = True
                            
                            response = "I've loaded product information. Please ask your question again."
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error loading products: {e}")
                        response = "I'm having trouble accessing the product database. Please try again later."
                else:
                    # Recreate vector store from cloud collection
                    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
                    qdrant_url = os.getenv("QDRANT_URL")
                    qdrant_api_key = os.getenv("QDRANT_API_KEY")
                    
                    if not qdrant_url or not qdrant_api_key:
                        response = "Configuration error: Qdrant credentials not found"
                    else:
                        client = QdrantClient(
                            url=qdrant_url,
                            api_key=qdrant_api_key
                        )
                        vector_store = Qdrant(
                            client=client,
                            collection_name=collection_name,
                            embeddings=embeddings
                        )
                        
                        # Run async operation in a synchronous context
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(generate_response(user_query, vector_store, text_chunks_sample))
                        loop.close()

                session['messages'].append({
                    'sender': 'bot',
                    'text': response,
                    'timestamp': datetime.now().strftime("%I:%M %p")
                })
                
                session.modified = True
                return jsonify({'bot_response': response})
                
            except Exception as e:
                logger.error(f"Error in prompt route: {e}", exc_info=True)
                return jsonify({'error': 'An error occurred processing your request'}), 500

    return render_template("prompt.html", messages=session.get('messages'))

@app.route("/clear-session", methods=["POST"])
def clear_session():
    # Clear the session but keep login status
    logged_in = session.get('logged_in')
    user = session.get('user')
    session.clear()
    
    if logged_in:
        session['logged_in'] = logged_in
        session['user'] = user
    
    init_messages()
    return jsonify({"status": "success"})

async def generate_response(user_query: str, vector_store: Any, text_chunks: List[str]) -> str:
    try:
        logger.info(f"Received query: {user_query}")
        
        if not text_chunks:
            return "Please wait while I load product information. Try asking your question again in a moment."
            
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            texts=text_chunks
        )
        
        docs = await hybrid_retriever.get_relevant_documents(user_query)
        
        if not docs:
            return "I apologize, but I couldn't find specific information about that in my current knowledge. Would you like to ask about something else?"
            
        prompt_template = """
        You are a helpful assistant. Use the provided content to answer questions accurately and professionally.

        Current Question: {question}
        Available Content: {context}

        Instructions:
        1. If the user asks about multiple products, mix them and provide the answer.
        2. Use the provided content to answer questions comprehensively and don't mention "based on the provided content" in whole conversation.
        3. If specific information isn't available, politely say so
        4. Maintain a friendly, helpful tone 
        5. Reference relevant content when appropriate and don't say "Based on the content" in the whole conversation.
        6. Stay focused on the information provided in the content
        7. Recommend the products with complete information and highlight them.
        8. If user query with abbreviations, expand them and provide the answer.
        10. If the content doesn't directly address the question, respond with: "I apologize, but that information isn't available in my current knowledge. Would you like to ask anything else?"

        Answer:
        """
        
        context = "\n".join([doc.page_content for doc in docs])
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash",
            temperature=0.6,
            top_p=0.95,
            top_k=40
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt
        )
        response = chain.run(
            input_documents=docs,
            question=user_query,
            context=context
        )
        
        return response.strip()

    except Exception as e:
        logger.error(f"Error in generate_response: {e}", exc_info=True)
        return "I apologize, but I encountered an error processing your request. Would you like to try asking in a different way?"

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if not all([email, password, confirm_password]):
            return render_template("signup.html", error="All fields are required")
        
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")
            
        # In a real application, you would:
        # 1. Hash the password
        # 2. Store user data in a database
        # 3. Send verification email
        
        return redirect(url_for('login'))
        
    return render_template("signup.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form['email']
        
        if not email:
            return render_template("forgot_password.html", error="Email is required")
            
        # In a real application, you would:
        # 1. Verify email exists in database
        # 2. Generate password reset token
        # 3. Send reset email
        
        return render_template("forgot_password.html", 
                             message="If an account exists with this email, "
                                    "you will receive password reset instructions.")
        
    return render_template("forgot_password.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

def create_app():
    try:
        initialize_vertex_ai()
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8081, debug=False, use_reloader=False)