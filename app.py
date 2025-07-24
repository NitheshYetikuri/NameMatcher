import streamlit as st
from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

class NameMatcher:
    def __init__(self, collection_name: str = "name_collection"):
        self.collection_name = collection_name
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH")

        if self.google_api_key is None:
            st.error("Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()
        if self.chroma_db_path is None:
            st.error("Error: CHROMA_DB_PATH not found in environment variables. Please set it in your .env file.")
            st.stop()

        self.vector_store = None
        self.chroma_client = None
        self.embedding_model = None
        self.google_ef = None

    def _initialize_vector_store(self):
        try:
            self.google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                model_name="models/embedding-001", api_key=self.google_api_key
            )
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=self.google_api_key
            )

            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.google_ef
            )

            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
            st.success(f"Vector store for collection '{self.collection_name}' initialized successfully.")
            return True
        except Exception as e:
            st.error(f"Error: Failed to initialize vector store for '{self.collection_name}'. Details: {e}")
            return False

    @staticmethod
    def _preprocess_text(text_list: list[str]) -> list[str]:
        if not isinstance(text_list, list):
            raise TypeError("Input for _preprocess_text must be a list of strings.")
        
        processed = []
        for text in text_list:
            if not isinstance(text, str):
                st.warning(f"Skipping non-string element during preprocessing: {text}")
                continue
            processed.append(text.strip().lower())
        return processed

    def add_names(self, names: list[str]):
        if self.vector_store is None:
            st.warning("Vector store not initialized. Cannot add names.")
            return

        try:
            processed_names = self._preprocess_text(names)
            self.vector_store.add_texts(texts=processed_names)
            st.success(f"Added {len(processed_names)} names to the vector store.")
        except Exception as e:
            st.error(f"Error: Failed to add names to vector store. Details: {e}")
 
    def find_similar_names(self, query: str, k: int = 10) -> list:
        if self.vector_store is None:
            st.warning("Vector store not initialized. Cannot perform search.")
            return []
        if not isinstance(query, str) or not query.strip():
            st.warning("Query must be a non-empty string.")
            return []

        try:
            retriever = self.vector_store.similarity_search_with_relevance_scores(query=query, k=k)
            st.info(f"Found {len(retriever)} similar names for query: '{query}'")
            return retriever
        except Exception as e:
            st.error(f"Error: Failed to perform similarity search for '{query}'. Details: {e}")
            return []

    def delete_collection(self):
        if self.chroma_client is None:
            st.error("ChromaDB client not initialized. Cannot delete collection.")
            return

        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.vector_store = None
            self.collection = None
            st.success(f"Collection '{self.collection_name}' successfully deleted.")
        except Exception as e:
            st.error(f"Error: Failed to delete collection '{self.collection_name}'. Details: {e}")

    def get_collection(self, names_to_add: list[str] = None):
        if self.chroma_client is None:
            self._initialize_vector_store()
            if self.chroma_client is None:
                st.error("ChromaDB client could not be initialized.")
                return False

        try:
            if self._initialize_vector_store():
                 st.info(f"Attempted to get/create collection '{self.collection_name}'. "
                         "If it existed, it's now active. If not, it was created.")
                 
                 if names_to_add and self.vector_store.get(ids=["Geetha"]).get('ids') == []: # Check if data exists
                     st.session_state.name_matcher.add_names(names_to_add) # Add names only if collection was truly empty
                 elif names_to_add:
                     st.info("Collection already contains data. Skipping adding sample names.")

                 return True
            else:
                 return False
        except Exception as e:
            st.error(f"Error trying to get collection '{self.collection_name}'. Details: {e}")
            return False

st.set_page_config(page_title="Name Matcher App", layout="centered")

st.title("💡 Name Matcher & Chat App")
st.write("Find similar names using AI embeddings and manage your vector database.")

if 'name_matcher' not in st.session_state:
    st.session_state.name_matcher = NameMatcher()

st.header("🗃️ Collection Management")
current_collection_name = st.text_input("Enter Collection Name (e.g., 'my_names'):", value="test", key="collection_name_input")

col1, col2 = st.columns(2)

# Define names_dataset globally so it's accessible
names_dataset = [
    "Geetha", "Gita", "Gitu", "Githa", "Keetha", "Meetha", "Seetha", "Heetha", "Sheetal", "Geeta",
    "John", "Jon", "Jhon", "Jonn", "Joan", "Jane", "Jayne", "Jan", "Janne", "Johnny",
    "Michael", "Mike", "Micheal", "Mikel", "Micael", "Mykel", "Miguel", "Michele", "Michaela", "Micaella",
    "Sarah", "Sara", "Saara", "Serah", "Sarra", "Saraah", "Sarrah", "Sarita", "Saniya", "Saanvi",
    "David", "Dave", "Davide", "Davi", "Davie", "Davy", "Davidson", "Davis", "Dav", "Daan",
    "Priya", "Preeya", "Priyaa", "Priyah", "Priyo", "Priyaank", "Prisha", "Priyanka", "Priyadarshini", "Prithvi",
    "Rahul", "Raul", "Rahil", "Rahuul", "Rahu", "Rohul", "Raheel", "Rajan", "Ramesh", "Rakesh",
    "Chennai", "Chenai", "Chenna", "Chinai", "Channai", "Chennapattanam", "Madras", "Cheannai", "Sennai", "Shennai",
    "Mumbai", "Mumbay", "Bombay", "Mumbaai", "Mumbhai", "Bambai", "Mumbayyi", "Mumbey", "Mumba", "Mumby",
    "Delhi", "Dehli", "Del", "Dilli", "Delhe", "Dehlih", "Delhia", "Dilhi", "Dela", "Dhilli",
    "New York", "NYC", "Newyork", "Nu York", "Nyork", "New Yawk", "Newyork city", "Big Apple", "Nyu York", "New York City",
    "London", "Londun", "Londan", "Londin", "Lunden", "Lundun", "Londyn", "Lon", "Londo", "Lond"
]

with col1:
    if st.button("Create/Get Collection & Add Sample Names"):
        st.session_state.name_matcher = NameMatcher(collection_name=current_collection_name)
        # Pass the dataset here to be added after initialization
        st.session_state.name_matcher.get_collection(names_to_add=names_dataset)
        
with col2:
    if st.button("Delete Current Collection"):
        if st.session_state.name_matcher.collection_name == current_collection_name:
            st.session_state.name_matcher.delete_collection()
        else:
            temp_matcher_for_delete = NameMatcher(collection_name=current_collection_name)
            temp_matcher_for_delete.delete_collection()

st.header("💬 Name Search (Chat-like)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter a name or phrase to search for..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = st.session_state.name_matcher.find_similar_names(query=prompt, k=5)
        
        if results:
            response_content = "Here are the top matches:\n\n"
            best_match_doc, best_match_score = results[0]
            response_content += f"**Best Match:** '{best_match_doc.page_content}' (Score: {best_match_score:.4f})\n\n"
            response_content += "**Other Similar Matches:**\n"
            for i, (doc, score) in enumerate(results):
                response_content += f"- '{doc.page_content}' (Score: {score:.4f})\n"
        else:
            response_content = "No matches found for your query or an error occurred. Please try again or ensure the collection has names."
        
        st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()