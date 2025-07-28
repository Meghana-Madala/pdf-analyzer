import streamlit as st
import os
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

#load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

# Initilize Azure Open AI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

with st.sidebar:
    st.title("PDF Chat Bot")
    st.markdown('''This isa chat bot that helps to analyze the uploaded large pdf file and answers the questions in humanized form using Large Language Model (LLM)''')
    st.write("PDF Analyzer")


def get_embedding_dimension():
    # This creates embedding using the ChatBot using ada-200 model
    return 1536

dimension = get_embedding_dimension()

def main():
    st.header("Query")

    pdf = st.file_uploader("Upload PDF file", type="pdf", key="pdf_uploader")
    st.write("Uploaded PDF - ", pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.faiss"):
            with open(f"{store_name}.faiss", "rb") as f:
                index = pickle.load(f)
            st.write("loaded from disk")
        else:
            embeddings = []
            for chunk in chunks:
                response = client.embeddings.create(
                    input=chunk,
                    model="ChatBot"
                )
                
                embedding.append(response.data[0].embedding)

                #Ensure all embeddings have the correct dimension
                embeddings = [emb[:dimension] for emb in embeddings if len(emb) > dimension]

                #Add embeddings to FAISS index
                if embeddings:
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings))
                else:
                    st.warning("No valid embeddings found")

                # Save FAISS index
                with open(f"{store_name}.faiss", "wb") as f:
                    pickle.dump(index, f)
                st.write("Embedded")

            def answer_question(question):
                query_embedding = client.embeddings.create(
                    input=question,
                    model="ChatBot"
                ).data[0].embedding

                print("========")
                print(f"FAISS index dimension: {index.d}")
                print("===========")
                print(f"Query Embedding shape: {np.array(query_embedding).shape}")

                #Ensure query embedding matches FAISS index dimension
                if isinstance(query_embedding, list):
                    query_embedding = np.array(query_embedding)

                if query_embedding.shape[0] != index.d:
                    print(f"Mismatch detected. Query embedding shape: {query_embedding.shape}")
                    print(f"FAISS index dimension: {index.d}")

                    #Truncate the query embedding if necessary
                    if query_embedding.shape[0] > index.d:
                        query_embedding = query_embedding[:index.d]
                        print(f"Truncated query embedding to {index.d} dimension")
                    else:
                        # Pad query if necessary
                        padding = np.zeros((index.d - query_embedding.shape[0],))
                        query_embedding = np.concatenate([query_embedding, padding])
                        print(f"Padded query embedding to {index.d} dimension")

                top_k = 4
                distances, indices = index.search(np.array([query_embedding]), k=top_k)

                relevant_paragraphs_with_score = []
                for idx in indices[0]:
                    passage = chunks[idx]
                    score = np.exp(-distances[0][indices[0].tolist().index(idx)])
                    relevant_paragraphs_with_score.append((passage, score))

                return relevant_paragraphs_with_score
            
            user_input = st.text_area("Enter your question:", key="question")

            if st.button("Ask Question"):
                result = answer_question(user_input)

                # Generate responses using Azure OpenAI LLM
                prompt = f"""
                    Give short answer to the question based on the given relevant passages
                    
                    Question:
                    {user_input}

                    Relevant Passages:
                    {result}
                """

                st.write(f"Answer to the Question : {user_input} is ")

                response = client.chat.completions.create(
                    model="gpt-4-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=400
                )

                response_dict = response.choices[0].message.model.model_dump()

                st.write(response_dict)



if __name__ == "__main__":
    main()