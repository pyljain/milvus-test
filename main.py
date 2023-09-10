from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import PyPDF2
import hashlib

def read_pdf_to_text(pdf_path):
    # Initialize a PDF file reader
    pdfFileObj = open(pdf_path, 'rb')
    
    # Create a PDF file reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    # Initialize text variable
    text = ""
    
    # Loop through each page
    for page_num in  range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[page_num]
        text += pageObj.extract_text()
        
    # Close the PDF file
    pdfFileObj.close()
    
    return text

def generate_sentence_embeddings(text):
    # Initialize SentenceTransformer model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    # Split text into sentences (assuming that each sentence ends with a period)
    sentences = text.split('. ')
    
    # Generate embeddings
    embeddings = model.encode(sentences)
    
    return sentences, embeddings

def main():
    connections.connect(
        alias="default",
        user='root',
        password='Milvus',
        host='localhost',
        port='19530',
    )

    text = read_pdf_to_text("./ID22-CEO-Transcript.pdf")
    sentences, embeddings = generate_sentence_embeddings(text)

    for sent, emb in zip(sentences, embeddings):

        # Calculater md5 of the sent
        sent_hash = hashlib.md5(sent.encode("utf-8")).hexdigest()

        # Insert into milvus
        data = {
            "embedding_id": sent_hash,
            "embedding": emb,
            "text": sent
        }

        default_collection = Collection("pdf_search")  
        resp = default_collection.insert(data)
        print(resp)


        print(f"Sentence: {sent}")
        print(f"Embedding: {emb[:5]}...")  # Print first 5 dimensions for brevity


def create_collection():

    connections.connect(
        alias="default",
        user='root',
        password='Milvus',
        host='localhost',
        port='19530',
    )

    collection_name = "pdf_search"

    utility.drop_collection(collection_name)

    embedding_id = FieldSchema(
        name="embedding_id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        max_length=50,
    )

    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=768
    )

    text = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=1000,
    )
    
    schema = CollectionSchema(
        fields=[embedding_id, embedding, text],
        description="PDF Search",
        enable_dynamic_field=True
    )

    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2
    )

    print(collection)


if __name__ == '__main__':
    main()