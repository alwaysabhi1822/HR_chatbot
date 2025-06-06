from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
import logging
import mysql.connector
from mysql.connector import pooling
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional
from uuid import uuid4
from datetime import datetime, timedelta
import bcrypt
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
load_dotenv()

logging.basicConfig(level=logging.INFO)

class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': '182209',
        'database': 'hr_chatbot'
    }
    STATIC_VECTOR_PATH = Path("static_vectors.pkl")
    DATA_FILE = Path("total_data.txt")
    SESSION_TIMEOUT = timedelta(minutes=30)

# Database connection pooling
db_pool = pooling.MySQLConnectionPool(
    pool_name="hr_pool",
    pool_size=5,
    **Config.DB_CONFIG
)

@contextmanager
def db_connection() -> Generator[mysql.connector.connection.MySQLConnection, None, None]:
    conn = db_pool.get_connection()
    try:
        yield conn
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'history': [],
            'user_info': {'role': 'guest'}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        session = self.sessions.get(session_id)
        if session and datetime.now() - session['last_activity'] < Config.SESSION_TIMEOUT:
            session['last_activity'] = datetime.now()
            return session
        if session:
            del self.sessions[session_id]
        return None
    
    def add_message(self, session_id: str, message: Dict):
        if session := self.get_session(session_id):
            session['history'].append(message)
    
    def update_user_info(self, session_id: str, info: Dict):
        if session := self.get_session(session_id):
            session['user_info'].update(info)
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()

def split_text_to_documents(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def initialize_static_vectorstore():
    if Config.STATIC_VECTOR_PATH.exists():
        try:
            with open(Config.STATIC_VECTOR_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading vectors: {e}")
    
    if not Config.DATA_FILE.exists():
        raise FileNotFoundError(f"Data file {Config.DATA_FILE} not found")
    
    with open(Config.DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = split_text_to_documents(text)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    with open(Config.STATIC_VECTOR_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama3-70b-8192")

def get_conversation_prompt(role: str):  # Changed to accept role
    base_system = """You are HR Assistant. Follow these STRICT rules:"""
    
    employee_prompt = f"""{base_system}
User Role: Employee
User Name: {{user_info_name}}

RULES FOR EMPLOYEES:
1. Use ONLY the following employee data:
{{context}}

2. For numbers:
- Use EXACT values from data
- NO calculations/estimations
- If missing, say "not available"

3. For medical info:
- Use EXACT blood group from data

4. Never refer to static HR documents
5. For salary slips:
- List EXACT links from data
"""
    
    guest_prompt = f"""{base_system}
User Role: Guest

RULES FOR GUESTS:
1. Answer only general HR questions
2. Redirect specific queries to HR
3. Never disclose employee data
Context: {{context}}"""
    
    return ChatPromptTemplate.from_messages([
        ("system", employee_prompt if role == "employee" else guest_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


static_vectorstore = initialize_static_vectorstore()
retriever = static_vectorstore.as_retriever()

chains = {
    "guest": create_retrieval_chain(
        create_history_aware_retriever(llm, retriever, get_conversation_prompt("guest")),
        create_stuff_documents_chain(llm, get_conversation_prompt("guest"))
    ),
    "employee": create_stuff_documents_chain(
        llm,
        get_conversation_prompt("employee")
    )
}

app = FastAPI()

# Add this under your existing CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class EmailRequest(BaseModel):
    email: str

class PasswordRequest(BaseModel):
    email: str
    password: str

class Query(BaseModel):
    query: str
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    response: str
    is_new_session: bool = False

class LoginResponse(BaseModel):
    session_id: str
    message: str
    user_data: Optional[Dict] = None
    success: bool = False

@app.post("/create-session")
async def create_guest_session():
    session_id = session_manager.create_session()
    return {"session_id": session_id}

# Update the ask endpoint with better error handling
@app.post("/ask", response_model=SessionResponse)
async def ask(query: Query):
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Empty query")

        # Validate session
        session = session_manager.get_session(query.session_id)
        if not session:
            return JSONResponse(
                status_code=401,
                content={"response": "Session expired. Please log in again."}
            )

        # Get user context
        user_info = session['user_info']
        context = []
        if user_info.get('role') == "employee":
            context = [Document(
                page_content=f"STRICT EMPLOYEE DATA (USE ONLY THIS):\n{user_info['employee_data']}"
            )]
        else:
            context = static_vectorstore.similarity_search(query.query)

        # Prepare LLM inputs
        chat_history = [
            AIMessage(content=msg['assistant']) if 'assistant' in msg else
            HumanMessage(content=msg['user'])
            for msg in session['history']
        ]

        inputs = {
            "input": query.query,
            "chat_history": chat_history,
            "context": context,
            "user_info_role": user_info.get('role', 'guest'),
            "user_info_name": user_info.get('name', 'Guest')
        }

        # Select chain
        chain = chains["employee"] if user_info.get('role') == "employee" else chains["guest"]
        #print("FINAL INPUTS TO PROMPT:", inputs)
        response = await chain.ainvoke(inputs)
        
        # Handle different response formats
        if isinstance(response, dict):
            answer = response.get("answer", "").strip()
        else:
            answer = str(response).strip()

        # Store interaction
        session_manager.add_message(query.session_id, {
            "user": query.query,
            "assistant": answer,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "session_id": query.session_id,
            "response": answer,
            "is_new_session": False
        }

    except Exception as e:
        logging.error(f"Full error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"response": f"Error processing request: {str(e)}"}
        )

@app.post("/check-email")
async def check_email(request: EmailRequest):
    try:
        with db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    "SELECT email FROM employee_auth WHERE email = %s", 
                    (request.email,)
                )
                return {"exists": cursor.fetchone() is not None}
    except Exception as e:
        logging.error(f"Email error: {e}")
        raise HTTPException(status_code=500, detail="Email check failed")

@app.post("/verify-password", response_model=LoginResponse)
async def verify_password(request: PasswordRequest):
    try:
        with db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    "SELECT password_hash FROM employee_auth WHERE email = %s", 
                    (request.email,)
                )
                result = cursor.fetchone()

                if not result:
                    return LoginResponse(success=False, message="Email not found")

                if not bcrypt.checkpw(
                    request.password.encode('utf-8'),
                    result['password_hash'].encode('utf-8')
                ):
                    return LoginResponse(success=False, message="Invalid password")

                session_id = session_manager.create_session()
                cursor.execute(
                    "SELECT employee_json FROM employee_data WHERE email = %s", 
                    (request.email,)
                )
                user_data = cursor.fetchone()
                #print(f"\n🔥 RAW DATA FETCHED FOR {request.email}:")
                #print(user_data['employee_json'])  # This will show EXACTLY what's being loaded
                
                parsed_data = None
                if user_data and user_data['employee_json']:
                    try:
                        parsed_data = json.loads(user_data['employee_json'])
                        formatted_data = "\n".join(
                        [f"• {k.replace('_', ' ').title()}: {v}" 
                        for k, v in parsed_data.items() 
                        if k not in ['password', 'history']]  # Filter sensitive fields
                        )

                        
                        session_manager.update_user_info(session_id, {
                            "email": request.email,
                            "role": "employee",
                            "name": parsed_data.get("name", "Employee"),
                            "employee_data": formatted_data
                        })
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON error: {e}")

                return LoginResponse(
                    session_id=session_id,
                    message="Login successful",
                    user_data=parsed_data,
                    success=True
                )
    except Exception as e:
        logging.error(f"Password error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/logout")
async def logout(session_id: str):
    session_manager.clear_session(session_id)
    return {"status": "success"}

@app.get("/")
async def serve_ui():
    try:
        with open("chat.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Chat UI not found</h1>",
            status_code=404
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)