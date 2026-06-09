# CDISC SDTM AI Assistant & Semantic Metadata Router (RAG & Streamlit)

## 📌 Business Case & Problem Statement

**⚠️ Note:** This project is currently a *Work in Progress (WIP)* as additional domain mappings and advanced validation checks continue to be integrated.

In clinical trials, mapping raw operational data to CDISC standards is a meticulous, high-stakes task for statistical and clinical programmers. Navigating thousands of pages across the SDTM Implementation Guide (SDTMIG) while simultaneously checking strict, granular definitions in the NCI Controlled Terminology (CT) spreadsheets often leads to data friction. Traditional search tools rely on simple keyword matches that fail to grasp clinical context, causing programmers to lose time looking up variables, checking acronym synonyms, or confirming whether a codelist is extensible. Relying on manually verified compliance or static lookups introduces a risk of structural alignment errors that can delay regulatory submissions.

**The Solution:** This project builds an intelligent compliance and mapping assistant utilizing **Generative AI, Large Language Models (LLMs), and Machine Learning components** via a **Retrieval-Augmented Generation (RAG)** architecture. Grounded in authoritative clinical sources (such as the SDTMIG v3.4 PDF and the CDISC Controlled Terminology Excel spreadsheets), the system bypasses the limitations of traditional search. 

Instead of relying solely on statistical vector matching, which can cause collisions among short clinical acronyms, I engineered a **Hybrid Deterministic Metadata Router**. The engine intercepts user queries, applies natural language tokenization to strip filler words, and scans custom multi-field metadata tags (`submission_value`, `codelist_code`, `codelist`, and `synonyms`) using high-precision database filters. If a programmer asks for short acronyms (e.g. `LOC`), exact NCI codes (e.g. `C74456`), or full descriptive phrases (e.g. `Anatomical Location`), the router locks directly onto the correct global codelist family with 100% precision. If no direct metadata match is found, the system smoothly switches to a vector-space proximity fallback to retrieve the most contextually relevant pages. The extracted data is then fed to the **Gemini 2.5 Flash** model, which generates a concise, validated response with specific document sources and page citations.

---

## ⚙️ Technical Blueprint & Engine Features

I engineered this project using robust data engineering principles, semantic text representation learning, and custom routing logic:

* **Hybrid Metadata-Semantic Routing:** Combines the flexibility of vector proximity search with strict database filtering logic. It uses a custom implementation of ChromaDB's logical `$or` operator to query across multiple custom metadata fields simultaneously, preventing vector collisions on short clinical strings.
* **Clinical Token & Multi-Word Phrase Extraction:** Features a text-processing pipeline that cleans punctuation and filters out natural language stop words. It automatically builds text word combinations (bigrams) to capture and evaluate literal multi-word descriptions like "Subject Status".
* **Vector Embeddings (Representation Learning):** Leverages deep-learning neural network architectures (`all-MiniLM-L6-v2`) via ChromaDB to map raw unstructured text into high-dimensional vector spaces, allowing the assistant to process conceptual and semantic clinical relationships.
* **Context Accumulation & Truncation Controls:** Dynamically adjusts the retrieval size based on how many tokens are in the query to handle comparative analysis across multiple codelists(e.g. comparing two codelists against eachother). It also uses specialized prompt engineering constraints to prevent output bloat, summarizing long-form data tables cleanly.

---

## 🛠️ Repository Architecture

```text
CDISC-AI-ASSISTANT/
│
├── embed_data.py        # Heavy text parsing, metadata mapping, and vector database indexing
├── app.py               # Live Streamlit application hosting the custom metadata routing and GenAI 
├── sdtmig_vector_db/    # Persistent local vector database files housing the embedded clinical context
├── requirements.txt     # Explicit package dependencies (including pysqlite3-binary for environment) 
└── README.md            # Technical documentation and project overview