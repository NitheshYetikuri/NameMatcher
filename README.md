# AI Assignment: Name Matching System

This project implements a name-matching system as part of Task 1 of the AI Assignment. It leverages **Google Generative AI embeddings** and **ChromaDB** to find similar names from a predefined dataset. The project provides both a **command-line interface (CLI)** for quick testing and a **Streamlit web application (UI)** for interactive use.

---

## Task 1: Get Matching Person Names

### Objective
To build a name-matching system that finds the most similar names from a dataset when a user inputs a name.

---

## Features

- **Data Preparation**: A predefined dataset of similar names (e.g., "Geetha", "Gita", "Gitu") is used.
- **Similarity Matching**: Utilizes Google Generative AI embeddings to convert names into vector representations and ChromaDB for efficient vector similarity search.
- **Output**:
  - The closest matching name with its similarity score.
  - A ranked list of other similar names with their scores.
- **Environment Variable Management**: Securely handles API keys and database paths using `.env` files.
- **Basic Error Handling**: Provides informative messages for common issues like missing environment variables or uninitialized vector stores.
- **Collection Management (UI)**: Allows users to create/get and delete ChromaDB collections directly from the web interface.
- **Interactive Chat (UI)**: A Streamlit-based chat interface to query the name matcher conversationally.

---

## Project Structure

```
NameMatcher/
├── .env                  # Environment variables (Google API Key, ChromaDB Path)
├── .gitignore            # Specifies files/directories to ignore in Git
├── main.py               # Command-Line Interface (CLI) application
├── app.py                # Streamlit Web Application (UI)
└── pyproject.toml        # Project dependencies managed by uv
```

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10+  
- `uv`: A fast Python package installer and resolver  
  Install it globally using:

```bash
pip install uv
```

---

## Setup Instructions

### 1. Place Project Files
Ensure `main.py`, `app.py`, `pyproject.toml`, and `.env` (to be created) are in the same directory.

### 2. Create and Activate Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add:

```
GOOGLE_API_KEY="YOUR_GOOGLE_GENERATIVE_AI_API_KEY_HERE"
CHROMA_DB_PATH="Path to store DB"
```

- Replace the API key with your actual key.
- Ensure the path exists or the app has permission to create it.

---



## How to Run

### 1. Command-Line Interface (CLI)

Run the CLI app:

```bash
uv run main.py
```

**Sample Interaction:**

```
enter user query : hi my name is geetha
```

**Expected Output:**

```
Vector store initialized successfully.
Added 48 names to the vector store.

--- Searching for: 'geetha' ---
Found 5 similar names for query: 'geetha'
Best Match: 'geetha' (Score: 0.5987)
Other Similar Matches:
  1. 'geetha' (Score: 0.5987)
  2. 'geetha' (Score: 0.5987)
  3. 'geetha' (Score: 0.5987)
  4. 'keetha' (Score: 0.5840)
  5. 'keetha' (Score: 0.5840)
```

---

### 2. Streamlit Web Application (UI)

Run the app:

```bash
streamlit run app.py
```

**Interaction:**

- **Collection Management**:
  - Enter a collection name (e.g., `test`)
  - Click **Create/Get Collection & Add Sample Names**
  - Optionally, click **Delete Current Collection**

- **Name Search**:
  - Enter a name in the chat input (e.g., `gita`)
  - View results in chat format
  - Click **Clear Chat History** to reset

**Expected Output:**

```
Vector store for collection 'test' initialized successfully.
Collection 'test' is now active.
Added 48 names to the vector store.
```

**Search Result Example:**

```
--- Searching for: 'geetha' ---
Found 5 similar names for query: 'geetha'
Here are the top matches:

Best Match: 'geetha' (Score: 0.5987)

Other Similar Matches:
- 'geetha' (Score: 0.5987)
- 'geetha' (Score: 0.5987)
- 'geetha' (Score: 0.5987)
- 'keetha' (Score: 0.5840)
- 'keetha' (Score: 0.5840)
```

---

## Deliverables

- `main.py`: CLI application
- `app.py`: Streamlit web application
- `.env`: Environment configuration file
- `pyproject.toml`: Dependency management
- `README.md`: Project documentation
