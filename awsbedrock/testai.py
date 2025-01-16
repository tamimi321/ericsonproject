import boto3
import botocore
import pandas as pd
import json
from decimal import Decimal
import streamlit as st
from time import sleep
# import openai

# Initialize AWS clients
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb",region_name="us-west-2")
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
EMBEDMODEL_ID = "anthropic.claude-v2"
# DynamoDB table name
DYNAMODB_TABLE = "DocumentEmbeddings"


# Step 1: Extract data from Excel using pandas
def extract_data_from_excel(bucket_name, file_key):
    # Download Excel file from S3
    s3.download_file(bucket_name, file_key, "C:\\Users\\AHMADT\\Documents\\tamz\\proj\\aws\\coba\\temp.xlsx")
    
    # Load Excel data into a pandas DataFrame
    df = pd.read_excel("C:\\Users\\AHMADT\\Documents\\tamz\\proj\\aws\\coba\\temp.xlsx")
    
    # Convert DataFrame rows into a list of dictionaries
    extracted_data = df.to_dict(orient="records")
    return extracted_data

def store_text_and_embeddings_in_dynamodb(document_id, text):
    # Generate embedding
    embedding = generate_embedding(text)
    if not embedding:
        return None
    
    # Convert float values to Decimal
    decimal_embedding = [Decimal(str(x)) for x in embedding]
    
    # Store the main document
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(
        Item={
            "DocumentID": document_id,
            "Content": text,
            "Type": "document",  # Marker to identify main document
            "ChunkIndex": -1  # Add this if ChunkIndex is part of the key
        }
    )
    
    # Store embedding chunks in separate items
    chunk_size = 5  # Even smaller chunk size
    for i in range(0, len(decimal_embedding), chunk_size):
        chunk = decimal_embedding[i:i + chunk_size]
        
        # Create a shorter document ID to save space
        chunk_id = f"{document_id[:10]}_{i//chunk_size}"  # Truncate document_id if needed
        
        # Store minimal information in each chunk
        table.put_item(
            Item={
                "DocumentID": chunk_id,
                "Content": text[:100] + "..." if len(text) > 100 else text,  # Truncate content
                "Type": "embedding",
                "ParentID": document_id,
                "ChunkIndex": i//chunk_size,
                "Values": chunk
            }
        )
    
    return True

def retrieve_text_and_embeddings_from_dynamodb():
    table = dynamodb.Table(DYNAMODB_TABLE)
    
    # Get all main documents
    response = table.scan(
        FilterExpression="#type = :doc_type",
        ExpressionAttributeNames={"#type": "Type"},
        ExpressionAttributeValues={":doc_type": "document"}
    )
    documents = response.get("Items", [])
    
    # For each document, retrieve and reconstruct its embedding
    for doc in documents:
        doc_id = doc["DocumentID"]
        
        # Get all chunks for this document
        chunk_response = table.scan(
            FilterExpression="#type = :emb_type AND #parent = :pid",
            ExpressionAttributeNames={
                "#type": "Type",
                "#parent": "ParentID"
            },
            ExpressionAttributeValues={
                ":emb_type": "embedding",
                ":pid": doc_id
            }
        )
        
        # Sort chunks by index and reconstruct embedding
        chunks = sorted(chunk_response["Items"], key=lambda x: x["ChunkIndex"])
        embedding = []
        for chunk in chunks:
            embedding.extend([float(x) for x in chunk["Values"]])
        
        doc["Embedding"] = embedding
    
    return documents

def search_similar_documents(query, documents):
    # Generate query embedding
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []
    
    # Calculate similarity (cosine similarity)
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude_v1 = sum(a * a for a in v1) ** 0.5
        magnitude_v2 = sum(b * b for b in v2) ** 0.5
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0
        return dot_product / (magnitude_v1 * magnitude_v2)

    # Find similar documents
    similarities = [
        {
            "DocumentID": doc["DocumentID"],
            "Content": doc["Content"],
            "Similarity": cosine_similarity(query_embedding, doc["Embedding"])
        }
        for doc in documents
        if "Embedding" in doc
    ]
    
    # Sort by similarity in descending order
    sorted_similarities = sorted(similarities, key=lambda x: x["Similarity"], reverse=True)
    return sorted_similarities[:3]  # Return top 3 results



def generate_embedding(text: str):
    try:
        body = json.dumps({
            "inputText": text
        })
        
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        return embedding
        
    except Exception as e:
        print(f"Error invoking Bedrock model for embeddings: {e}")
        return None

# Step 4: Search similar documents using embeddings
# def search_similar_documents(query, documents):
#     # Generate query embedding
#     # query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
#     query_embedding = generate_embedding(query)
#     # Calculate similarity (e.g., cosine similarity)
#     def cosine_similarity(v1, v2):
#         dot_product = sum(a * b for a, b in zip(v1, v2))
#         magnitude_v1 = sum(a * a for a in v1) ** 0.5
#         magnitude_v2 = sum(b * b for b in v2) ** 0.5
#         if magnitude_v1 == 0 or magnitude_v2 == 0:
#             return 0
#         return dot_product / (magnitude_v1 * magnitude_v2)

#     # Find the most similar document
#     similarities = [
#         {"DocumentID": doc["DocumentID"], "Content": doc["Content"], "Similarity": cosine_similarity(query_embedding, doc["Embedding"])}
#         for doc in documents
#     ]
#     # Sort by similarity in descending order
#     sorted_similarities = sorted(similarities, key=lambda x: x["Similarity"], reverse=True)
#     return sorted_similarities[:3]  # Return top 3 results


# Step 5: Query Amazon Bedrock with context
# def query_bedrock_with_context(context, user_question, model_id="anthropic.claude-v2", max_tokens=300):
#     prompt = f"""
#     You are a helpful assistant. Use the following context to answer the question:

#     Context: {context}

#     Question: {user_question}
#     """
#     response = bedrock.invoke_model(
#         modelId=model_id,
#         body={
#             "prompt": prompt,
#             "max_tokens": max_tokens,
#         }
#     )
#     return response["body"]["completions"][0]["text"]

def query_bedrock_with_context(context, user_question, model_id="amazon.titan-text-express-v1", max_tokens=300):
    prompt = f"""Use the following context to answer the question:

    Context: {context}

    Question: {user_question}"""
    
    try:
        # Titan model expects a different request body structure
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": []
            }
        })

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        # Titan model returns results in a different format
        return response_body.get('results', [{}])[0].get('outputText', '')
            
    except botocore.exceptions.ClientError as error:
        print(f"Error calling Bedrock: {error}")
        return "I apologize, but I encountered an error processing your request."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "I apologize, but something went wrong."



# Main function to integrate all steps
def chatbot(bucket_name, file_key, document_id_prefix, user_question):
    # Step 1: Extract data from Excel
    # extracted_data = extract_data_from_excel(bucket_name, file_key)

    # # Step 2: Store each row of data and its embedding in DynamoDB
    # for idx, row in enumerate(extracted_data):
    #     # Convert row dictionary to a string for embedding
    #     row_text = " | ".join(f"{key}: {value}" for key, value in row.items())
    #     document_id = f"{document_id_prefix}_{idx}"  # Unique ID for each row
    #     store_text_and_embeddings_in_dynamodb(document_id, row_text)

    # Step 3: Retrieve all documents from DynamoDB
    documents = retrieve_text_and_embeddings_from_dynamodb()
    print("done retrieve data from dynamoDB")
    # Step 4: Find similar documents
    similar_documents = search_similar_documents(user_question, documents)

    # Retrieve context from the most similar document
    if similar_documents:
        context = similar_documents[0]["Content"]  # Use the most similar document's content
    else:
        context = "No relevant documents found."

    # Step 5: Query Amazon Bedrock with context
    response = query_bedrock_with_context(context, user_question)
    return response

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_chatbot():
    # Set page configuration
    st.set_page_config(page_title="AWS Chatbot", page_icon="🤖")
    st.title("AWS Document Assistant 🤖")
    
    # Add description
    st.markdown("""
    This chatbot helps you find information from your AWS documents. 
    Ask any question about your data!
    """)

def display_chat_history():
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_chatbot():
    # Set page configuration
    st.set_page_config(page_title="AWS Chatbot", page_icon="🤖")
    st.title("AWS Document Assistant 🤖")
    
    # Add description
    st.markdown("""
    This chatbot helps you find information from your AWS documents. 
    Ask any question about your data!
    """)

def display_chat_history():
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_user_input(user_input):
    # Configuration (you can modify these as needed)
    bucket_name = "wavenetbucket"
    file_key = "capacity.xlsb"
    document_id_prefix = "excel_row"
    
    # Get chatbot response using your existing function
    response = chatbot(bucket_name, file_key, document_id_prefix, user_input)
    return response

def main():
    initialize_chatbot()
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with typing indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get response from your chatbot
            assistant_response = process_user_input(prompt)
            
            # Simulate typing with a simple animation
            for chunk in assistant_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                sleep(0.05)
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()


# # Example usage
# if __name__ == "__main__":
#     # Configuration
#     bucket_name = "wavenetbucket"
#     file_key = "capacity.xlsb"
#     document_id_prefix = "excel_row"
#     user_question = "What is the link with the worst bandwith ?"

#     # Run the chatbot
#     chatbot_response = chatbot(bucket_name, file_key, document_id_prefix, user_question)
#     print("Chatbot Response:", chatbot_response)
