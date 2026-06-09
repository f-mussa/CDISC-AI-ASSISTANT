__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import chromadb
from google import genai

# Step 1: Config web page layout
st.set_page_config(
    page_title="CDISC AI Assistant",
    page_icon="🧬"
)

# Automated Cloud Initialization Condition
# Triggers only if the app is launched on the cloud and the database folder is missing
DB_DIR = "./sdtmig_vector_db"
if not os.path.exists(DB_DIR):
    with st.spinner("🚀 Initializing system container: Building CDISC Vector Database on the cloud. This may take a few minutes..."):
        try:
            # Import your parsing logic dynamically
            from embed_data import extract_and_index_pdf, extract_and_index_xls
            
            # Define your source file paths (Make sure these source files are pushed to GitHub)
            pdf_source = "./cdisc_docs/SDTMIG v3.4-FINAL_2022-07-21.pdf" 
            excel_source = "./cdisc_docs/SDTM Terminology.xls"
            
            # Run the build processes sequentially on Streamlit's server
            extract_and_index_pdf(pdf_source)
            extract_and_index_xls(excel_source)
            st.success("🎉 Database successfully generated and optimized on server instance!")
        except Exception as e:
            st.error(f"Failed to compile database on initial boot: {e}")
            st.stop()

# Step 2: setup Gemini API (fetch API Key from Streamlit)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Missing Gemini API Key! Please set it in your environment variables.")
    st.stop()
client = genai.Client(api_key=GEMINI_API_KEY)

# Step 3: Connect to both SDTMIG/CT vector DBs
chroma_client = chromadb.PersistentClient(path=DB_DIR)
pdf_collection = chroma_client.get_collection(name='sdtmig_v3_4')
ct_collection = chroma_client.get_collection(name='cdisc_ct')

# Step 4: Render UI Titles and Elements
st.title("🧬 CDISC SDTM IG Assistant")
st.write("Ask any questions regarding SDTMIG and CDISC CT(e.g. domains, codelists, etc.)")

# Step 5: Get user question and generate answer using Gemini
# User input
user_query = st.text_input("Enter your question:", placeholder="e.g., What are the valid submission values for the NY codelist?")

if user_query:
    with st.spinner("Searching all CDISC assets and generating answer..."):

        # 1. Query SDTMIG vector DB for top 3 relevant context chunks
        pdf_results = pdf_collection.query(query_texts=[user_query], n_results=3)
        pdf_chunks = pdf_results['documents'][0] if pdf_results['documents'] else [] # Extract text chunks from results

        # 2. Query CT vector DB to find the parent codelist group
        discovered_codelist = None
        ct_chunks = []

        # Strip punctuation and common filler/stop words to isolate clinical tokens 
        filler_words = {"WHAT", "IS", "THE", "FOR", "A", "VALID", "SUBMISSION", "VALUES", "CODELIST", "ARE", "OF", "SHOW", "LIST", "WHICH", "IN"}
        raw_words = [word.strip("?,.:;\"'()").upper() for word in user_query.split()]
        meaningful_tokens = [word for word in raw_words if word not in filler_words and len(word) > 1]

        # Reconstruct potential multi-word descriptive phrases (e.g., "ANATOMICAL LOCATION")
        # so we don't accidentally check words completely isolated if they belong together
        possible_phrases = []
        if len(meaningful_tokens) > 1:
            # Reconstruct the whole filtered chain
            possible_phrases.append(" ".join(meaningful_tokens))
            # Reconstruct pairs (bigrams) for longer text strings
            for idx in range(len(meaningful_tokens) - 1):
                possible_phrases.append(f"{meaningful_tokens[idx]} {meaningful_tokens[idx+1]}")

        # Combine everything we want to evaluate against our database keys
        search_candidates = meaningful_tokens + possible_phrases
        discovered_codelist_groups = set() # Store multiple distinct codelists if found
        for candidate in search_candidates:
            matched_records = ct_collection.get(
                where={"$or": [
                    {"submission_value": candidate},
                    {"codelist_code": candidate},
                    {"codelist": candidate}, # Matches things like "ANATOMICAL LOCATION" perfectly
                    {"synonyms": candidate}
                ]}
            )
            
            # If an exact metadata match hits, intercept the group's global Codelist Name
            if matched_records.get('metadatas') and len(matched_records['metadatas']) > 0:
                #discovered_codelist = matched_records['metadatas'][0].get('codelist')
                #break
                for meta in matched_records['metadatas']:
                    if meta and 'codelist' in meta:
                        discovered_codelist_groups.add(meta['codelist'])
        
        # 3. Extract documents for all identified codelist groups
        if discovered_codelist_groups:
            for group_name in discovered_codelist_groups:
                all_ct_records = ct_collection.get(
                    where={"codelist": group_name}
                )
                if all_ct_records.get("documents"):
                    ct_chunks.extend(all_ct_records['documents'])

        # Fallback: If no strict identifier token hit, fall back safely to standard semantic search nets
        else:
            ct_results = ct_collection.query(query_texts=[user_query], n_results=10)
            if ct_results.get('metadatas') and len(ct_results['metadatas']) > 0 and ct_results['metadatas'][0]:
                #fallback_name = ct_results['metadatas'][0][0].get('codelist')
                #if fallback_name:
                    #all_ct_records = ct_collection.get(where={"codelist": fallback_name})
                    #ct_chunks = all_ct_records['documents'] if all_ct_records.get("documents") else []

                # Extract all unique codelists found in the top semantic matches
                all_found_groups = list(set([meta['codelist'] for meta in ct_results['metadatas'][0] if meta and 'codelist' in meta]))

                # Dynamically cap maximum codelists (e.g. If user asked about 3 codelists, capture up to 3 groups. Default minumum to 2)
                max_fallback_targets = max(2, len(meaningful_tokens))
                fallback_groups = all_found_groups[:max_fallback_targets]

                for fallback_name in fallback_groups:
                      all_ct_records = ct_collection.get(where={"codelist": fallback_name})
                      if all_ct_records.get("documents"):
                            ct_chunks.extend(all_ct_records['documents'])

        context_text = "--- SDTM IMPLEMENTATION GUIDE CONTEXT ---\n"
        context_text += "\n\n".join(pdf_chunks)
        context_text += "\n\n--- CONTROLLED TERMINOLOGY CONTEXT ---\n"
        context_text += "\n\n".join(ct_chunks)

        # Construct specialized prompt
        prompt = f"""
        You are an expert Clinical Programmer and CDISC specialist. 
        Your task is to answer the user's question by cross-referencing BOTH the provided segments of the SDTM Implementation Guide (SDTMIG) and the CDISC Controlled Terminology (CT) records.
        
        CRITICAL INSTRUCTIONS FOR PROCESSING THE CONTROLLED TERMINOLOGY CONTEXT:
        1. The CT context contains an entire flattened codelist group pulled from an NCI Excel sheet.
        2. Identify the PARENT metadata row: This is the row where the "Submission Value" matches the shorthand codelist abbreviation (e.g., Submission Value: LOC for the Codelist Name: Anatomical Location). Use this to confirm the correct codelist.
        3. Identify the CHILD rows: These are all subsequent rows that share the same Codelist Name but have individual clinical terms (e.g., TRUNK, DIGIT, ARM, EYE) as their "Submission Value".
        
        4. EXECUTIVE DECISION FOR LONG CODELISTS (TRUNCATION RULE):
           - If a codelist contains more than 15 valid submission values, DO NOT list all of them.
           - Instead, list the first 10 to 15 terms alphabetically as examples to show the formatting standard.
           - Immediately following the truncated list, add a clear technical note instructing the programmer to reference the full CDISC Controlled Terminology spreadsheet/file or their internal metadata repository for the complete list of terms.
        
        Provide a highly professional, technical, and concise answer. Reference specific codelists, Codelist Codes (C-codes), or domain rules where applicable.
        If the context doesn't contain enough information, state that clearly. Do not make up or extrapolate values.
        
        Context provided:
        {context_text}
        
        Question: {user_query}
        
        Answer:
        """

        # Generate answer using Gemini AI
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        # Display answer
        st.markdown("### Assistant Response")
        st.write(response.text)

        # Display sources
        st.markdown("---")
        with st.expander("📚 View SDTMIG sources used for this answer"):
            st.markdown("#### Found in SDTMIG PDF:")
            for i, chunk in enumerate(pdf_chunks, 1):
                st.info(f"**Source {i}:** {chunk}")

            st.markdown("#### Found in Controlled Terminology Excel:")
            for i, chunk in enumerate(ct_chunks, 1):
                st.info(f"**Source {i}:** {chunk}")

    






