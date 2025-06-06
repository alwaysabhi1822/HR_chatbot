
# HR Chatbot Assistant


## Features ✨

- **Role-based authentication** (Employee/Guest)
- **Personalized responses** using employee-specific data
- **Natural language processing** via Groq's Llama3-70b
- **Conversation history** with session management
- **Document retrieval** using FAISS vector embeddings
- **Secure authentication** with bcrypt password hashing
- **Clean responsive UI** with animated chat interface

## Setup
1. Create MySQL database using `setup.sql`
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables in `.env`:

#DB_HOST=localhost

#DB_USER=root

#DB_PASSWORD=your_password

#DB_NAME=hr_chatbot

#GROQ_API_KEY=your_api_key

4. Run: `uvicorn app:app --reload`


