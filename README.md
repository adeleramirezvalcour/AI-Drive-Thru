Voice AI Drive-Thru System

An AI system that simulates a restaurant drive-thru experience using speech recognition, retrieval-augmented generation (RAG), intet classification, and LLM tool calling. This system uses Whisper hosted by Groq as well as Llama 3. 

Features:
- Whisper Large v3
- Retrieval-Augmented Generation (RAG)
- Intent classification
- FAISS vector similarity search
- LLM tool calling
- JSON outputs
- Guardrails for off-topic and invalid requests
- Text-to-speech response generation
- Menu knowledge through vector embeddings
- FastAPI service endpoint


Technologies:
- Python
- Llama 3
- Groq API
- Whisper Large v3
- FastAPI
- PyTorch
- Sentence Transformers
- FAISS
- pandas
- NumPy

System Architecture:
- Customer Speech
- Whisper STT
- Intent Classification
- FAISS Retrieval
- Menu Knowledge
- Llama 3 (Groq API)
- Tool Calling 
- Guardrails
- Response Generation
- Text-to-Speech


Example Interaction:
User: "I'd like a burger with fries"

System: "That's a Cheeseburger and Fries. Your total comes out to be $11.32. Would you like anything else today?"

Future Improvements:
- Cloud deployment
- Multi-turn converstaion memory
- Order tracking 
- Support of different quantities and modifications
- Website ordering interface
- Demo

