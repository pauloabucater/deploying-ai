from langchain.tools import tool
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field
# import sqlalchemy as sa
import pandas as pd
from dotenv import load_dotenv
from utils.logger import get_logger
import os
_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

# vector_db_client_url="http://localhost:8000"
# chroma = chromadb.HttpClient(host=vector_db_client_url)
chroma_folder = "assignment_chat/data/chromadb"
chroma = chromadb.PersistentClient(path=chroma_folder)

collection_name = "books_data"
collection = chroma.get_collection(name=collection_name, 
                                   embedding_function=OpenAIEmbeddingFunction(
                                       api_key = os.getenv("OPENAI_API_KEY"),
                                       model_name="text-embedding-3-small")
                                   )

# Ratings file: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data/code?select=Books_rating.csv
# Path: data/books_data.csv
# Rec count: 212,404
# Columns:
    # "Title",
    # "description",
    # "authors",
    # "image",
    # "previewLink",
    # "publisher",
    # "publishedDate",
    # "infoLink",
    # "categories",
    # "ratingsCount"


# Ratings file: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data/code?select=Books_rating.csv
# Path: data/books_rating.csv
# Rec count: 3,000,000
# Columns:
    # "Id",
    # "Title",
    # "Price",
    # "User_id",
    # "profileName",
    # "review_helpfulness",
    # "review_score",
    # "review_time",
    # "review_summary",
    # "review_text"


class BookData(BaseModel):
    """Structured book data response."""
    title: str = Field(None, description="The title of the book.")
    description: str = Field(None, description="The description of the book.")
    authors: str = Field(None, description="The authors of the book.")
    image: str = Field(None, description="The image URL of the book.")
    previewLink: str = Field(None, description="The preview link of the book.")
    publisher: str = Field(None, description="The publisher of the book.")
    publishedDate: str = Field(None, description="The published date of the book.")
    infoLink: str = Field(None, description="The info link of the book.")
    categories: str = Field(None, description="The categories of the book.")
    ratingsCount: int = Field(None, description="The ratings count of the book.")



@tool
def recommend_books(query: str, n_results: int = 1) -> list[BookData]:
    """Fetches book summary data based on the query. Returns n_results reviews."""
    _logs.debug(f"*** recommend_books - start: query={query}")
    recommendations = get_context(query, collection, n_results)
    _logs.debug(f"*** recommend_books - response: {recommendations}")
    return recommendations



def additional_details(title:str):
    _logs.debug(f"*** additional_details - start: title={title}")
    file = "assignment_chat/data/books_data.csv"
    df_books = pd.read_csv(file, encoding='utf-8')
    # filtered_df = df_books[df_books['Title'].str.contains(title, case=False, na=False)]
    filtered_df = df_books[df_books['Title'] == title]


    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        details = {
            "title": row['Title'],
            "authors": row['authors'],
            "description": row['description'],
            "publisher": row['publisher']
        }
        # _logs.debug(f'Found details for book {title}: {details}')
        return details
    else:
        _logs.warning(f'No details found for book: {title}')
        return {}


def get_title_from_custom_id(custom_id:str):
    _logs.debug(f"*** get_title_from_custom_id - start: custom_id={custom_id}")
    return custom_id.split('_')[0]

def get_context_data(query:str, collection:chromadb.api.models.Collection, top_n:int):
    _logs.debug(f"*** get_context_data - start: query={query}, top_n={top_n}")
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )
    context_data = []
    for idx, custom_id in enumerate(results['ids'][0]):
        title = get_title_from_custom_id(custom_id)
        # _logs.debug(f"*** get_context_data - processing: title={title}")
        details = additional_details(title)
        details['text'] = results['documents'][0][idx]
        context_data.append(details)
    return context_data

def get_context(query:str, collection:chromadb.api.models.Collection, top_n:int):
    _logs.debug(f"*** get_context - start: query={query}, top_n={top_n}")
    context_data = get_context_data(query, collection, top_n)
    recommendations = []
    if not context_data:
        return recommendations
    for item in context_data:

        rec = BookData(
            title=item.get('title', 'N/A'),
            authors=item.get('authors', 'N/A'),
            description=item.get('description', 'N/A'),
            publisher=item.get('publisher', 0.0)
        )
        recommendations.append(rec)
    return recommendations