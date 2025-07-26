import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional, Any
import logging
import google.generativeai as genai
from dataclasses import dataclass
import re
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

class MedicalRAGPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', gemini_api_key: Optional[str] = None):
        """
        Initialize the Medical RAG Pipeline with improved error handling
        """
        try:
            self.encoder = SentenceTransformer(model_name)
            self.index = None
            self.documents = []
            self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
            
            # Initialize Gemini
            if not gemini_api_key:
                gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            if not gemini_api_key:
                raise ValueError("Gemini API key not provided")
                
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Add exit patterns
            self.exit_patterns = {
                'quit', 'exit', 'bye', 'goodbye', 'terminate', 'end', 'sign off',
                'terminate the call', 'sign off', 'end call'
            }
            
            # Medical disclaimer
            self.medical_disclaimer = (
                "\nDisclaimer: This information is for educational purposes only. "
                "Please consult a healthcare professional for medical advice, diagnosis, or treatment."
            )
            
            logger.info("Successfully initialized Medical RAG Pipeline")
            
        except Exception as e:
            logger.error(f"Error initializing Medical RAG Pipeline: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load_diseases_data(self, diseases_path: str) -> List[Document]:
        """
        Load and process diseases data with retry mechanism
        """
        try:
            if not os.path.exists(diseases_path):
                raise FileNotFoundError(f"Diseases data file not found: {diseases_path}")
                
            with open(diseases_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for disease in data.get('diseases', []):
                try:
                    # Create comprehensive document for each disease
                    content_parts = []
                    
                    if 'tag' in disease:
                        content_parts.append(f"{disease['tag']}: {disease.get(disease['tag'], '')}")
                    if 'symptoms' in disease:
                        content_parts.append(f"Symptoms: {disease['symptoms']}")
                    if 'treatment' in disease:
                        content_parts.append(f"Treatment: {disease['treatment']}")
                    if 'types' in disease:
                        content_parts.append(f"Types: {disease['types']}")
                    if 'prevention' in disease:
                        content_parts.append(f"Prevention: {disease['prevention']}")
                    
                    content = "\n".join(content_parts)
                    
                    doc = Document(
                        content=content,
                        metadata={'tag': disease.get('tag', 'unknown')}
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing disease entry: {e}")
                    continue
            
            if not documents:
                raise ValueError("No valid disease documents were created")
                
            logger.info(f"Successfully loaded {len(documents)} disease documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading diseases data: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load_intents_data(self, intents_path: str) -> Dict:
        """
        Load intents data with retry mechanism
        """
        try:
            if not os.path.exists(intents_path):
                raise FileNotFoundError(f"Intents data file not found: {intents_path}")
                
            with open(intents_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'intents' not in data or not data['intents']:
                raise ValueError("No intents found in the data file")
                
            logger.info(f"Successfully loaded intents data with {len(data['intents'])} intents")
            return data
            
        except Exception as e:
            logger.error(f"Error loading intents data: {e}")
            raise

    def create_index(self, documents: List[Document]) -> None:
        """
        Create FAISS index with error handling
        """
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")
                
            # Convert documents to embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.encoder.encode(texts)
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            self.documents = documents
            
            logger.info(f"Successfully created FAISS index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def normalize_query(self, query: str) -> str:
        """
        Normalize query with improved handling
        """
        try:
            # Remove special characters and convert to lowercase
            query = re.sub(r'[^\w\s]', '', query.lower())
            
            # Handle common medical misspellings
            misspellings = {
                'canser': 'cancer',
                'diabetis': 'diabetes',
                'artritis': 'arthritis',
                'highbloodpressure': 'hypertension',
                'asma': 'asthma',
                'hart': 'heart',
                'stroke': 'stroke',
                'alzheimers': 'alzheimer',
                'highblood': 'hypertension',
                'colesterol': 'cholesterol',
                'anxiety': 'anxiety',
                'depresion': 'depression',
                'migrane': 'migraine'
            }
            
            # Split query into words and correct each word if it's misspelled
            words = query.split()
            corrected_words = [misspellings.get(word, word) for word in words]
            
            return ' '.join(corrected_words)
            
        except Exception as e:
            logger.error(f"Error normalizing query: {e}")
            return query

    def is_exit_request(self, query: str) -> bool:
        """
        Check if query is an exit request
        """
        try:
            normalized_query = query.lower().strip()
            return any(exit_pattern in normalized_query for exit_pattern in self.exit_patterns)
        except Exception as e:
            logger.error(f"Error checking exit request: {e}")
            return False

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search the index with improved error handling
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
                
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), k
            )
            
            # Return documents and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.documents):
                    results.append((self.documents[idx], float(distance)))
                else:
                    logger.warning(f"Invalid document index: {idx}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
    def generate_response(self, query: str, intents_data: Dict) -> str:
        """
        Generate response with retry mechanism and improved error handling
        """
        try:
            # Check for exit request
            if self.is_exit_request(query):
                return "Thank you for using our medical assistance service. Remember to consult healthcare professionals for medical advice. Goodbye!"

            # Normalize query
            normalized_query = self.normalize_query(query)
            
            # Search for relevant documents
            search_results = self.search(normalized_query)
            
            # Create context from search results
            context = "\n\n".join([doc.content for doc, _ in search_results])
            
            # Find matching intent
            matching_intent = None
            for intent in intents_data.get('intents', []):
                if any(pattern.lower() in normalized_query for pattern in intent.get('patterns', [])):
                    matching_intent = intent
                    break
            
            # Generate prompt
            prompt = f"""As a medical assistant, provide a clear and concise answer to the following question using the provided context. 
            Focus on accuracy and avoid repetition.

            Context:
            {context}

            Question: {query}

            Guidelines:
            1. Be concise and avoid repeating information
            2. If information is not available in the context, acknowledge that
            3. For symptoms or serious conditions, recommend consulting a healthcare provider
            4. Ensure the response is clear and well-structured
            5. If this is an emergency condition, emphasize seeking immediate medical attention
            """
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            final_response = response.text.strip()
            
            # Add intent response if available
            if matching_intent and 'responses' in matching_intent:
                intent_response = matching_intent['responses'][0]
                if intent_response not in final_response:
                    final_response = f"{final_response}\n\n{intent_response}"
            
            # Add medical disclaimer
            final_response += self.medical_disclaimer
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ("I apologize, but I'm having trouble generating a response. "
                   "Please try again or consult a healthcare professional for medical advice.")

    def process_audio_query(self, audio_handler: Any, audio_file: str, intents_data: Dict) -> Optional[Dict]:
        """
        Process audio query with improved error handling
        """
        try:
            # Transcribe audio to text
            query = audio_handler.transcribe_audio(audio_file)
            if not query:
                logger.error("Could not transcribe audio")
                return None

            # Generate text response
            text_response = self.generate_response(query, intents_data)

            # Convert response to audio
            audio_response = audio_handler.text_to_speech(text_response)
            if not audio_response:
                logger.error("Could not convert response to speech")
                return None

            return {
                'query': query,
                'text_response': text_response,
                'audio_response': audio_response
            }

        except Exception as e:
            logger.error(f"Error processing audio query: {e}")
            return None
