from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

class NameMatcher:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH")

        if self.google_api_key is None:
            print("Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            exit(1)
        if self.chroma_db_path is None:
            print("Error: CHROMA_DB_PATH not found in environment variables. Please set it in your .env file.")
            exit(1)

        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                model_name="models/embedding-001", api_key=self.google_api_key
            )
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=self.google_api_key
            )

            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, embedding_function=google_ef
            )

            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
            print("Vector store initialized successfully.")

        except Exception as e:
            print(f"Error: Failed to initialize vector store. Details: {e}")
            exit(1)

    @staticmethod
    def _preprocess_text(text_list: list[str]) -> list[str]:
        if not isinstance(text_list, list):
            raise TypeError("Input for _preprocess_text must be a list of strings.")
        
        processed = []
        for text in text_list:
            if not isinstance(text, str):
                print(f"Warning: Skipping non-string element during preprocessing: {text}")
                continue
            processed.append(text.strip().lower())
        return processed

    def add_names(self, names: list[str]):
        if self.vector_store is None:
            print("Error: Vector store not initialized. Cannot add names.")
            return

        try:
            processed_names = self._preprocess_text(names)
            self.vector_store.add_texts(texts=processed_names)
            print(f"Added {len(processed_names)} names to the vector store.")
        except Exception as e:
            print(f"Error: Failed to add names to vector store. Details: {e}")

    def find_similar_names(self, query: str, k: int = 10) -> list:
        if self.vector_store is None:
            print("Error: Vector store not initialized. Cannot perform search.")
            return []
        if not isinstance(query, str) or not query.strip():
            print("Warning: Query must be a non-empty string.")
            return []

        try:
            retriever = self.vector_store.similarity_search_with_relevance_scores(query=query, k=k)
            print(f"Found {len(retriever)} similar names for query: '{query}'")
            return retriever
        except Exception as e:
            print(f"Error: Failed to perform similarity search for '{query}'. Details: {e}")
            return []

if __name__ == "__main__":
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

    try:
        matcher = NameMatcher(collection_name="test")

        matcher.add_names(names_dataset)

        query_name = input("enter user query : ")
        print(f"\n--- Searching for: '{query_name}' ---")
        results = matcher.find_similar_names(query=query_name, k=5)

        if results:
            best_match_doc, best_match_score = results[0]
            print(f"Best Match: '{best_match_doc.page_content}' (Score: {best_match_score:.4f})")
            print("Other Similar Matches:")
            for i, (doc, score) in enumerate(results):
                print(f"  {i+1}. '{doc.page_content}' (Score: {score:.4f})")
        else:
            print("No matches found or an error occurred during search.")

    except SystemExit:
        print("\nApplication terminated due to a critical error. Please check the error messages above.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main program: {e}")