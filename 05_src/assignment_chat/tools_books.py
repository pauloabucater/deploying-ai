from langchain.tools import tool
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field
import sqlalchemy as sa
import pandas as pd
# from dotenv import load_dotenv
# from utils.logger import get_logger
import os
# _logs = get_logger(__name__)
# load_dotenv()
# load_dotenv(".secrets")


vector_db_client_url="http://localhost:8000"
chroma = chromadb.HttpClient(host=vector_db_client_url)
collection = chroma.get_collection(name="books", 
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


class BookReviewData(BaseModel):
    """Structured book review data response."""
    id: str = Field(None, description="The unique identifier for the book review.")
    title: str = Field(None, description="The title of the book.")
    price: str = Field(None, description="The price of the book.")
    user_id: str = Field(None, description="The user ID of the reviewer.")
    profileName: str = Field(None, description="The profile name of the reviewer.")
    review_helpfulness: str = Field(None, description="The helpfulness rating of the review.")
    review_score: float = Field(None, description="The score given by the reviewer.")
    review_time: str = Field(None, description="The time when the review was written.")
    review_summary: str = Field(None, description="A summary of the review.")
    review_text: str = Field(None, description="The full text of the review.")


@tool
def recommend_albums(query: str, n_results: int = 1) -> list[BookReviewData]:
    """Fetches music review data based on the query. Returns n_results reviews."""
    recommendations = get_context(query, collection, n_results)
    return recommendations


def additional_details(review_id:str):
    _logs.debug(f'Fetching additional details for review ID: {review_id}')
    engine = sa.create_engine(os.getenv("SQL_URL"))
    query = f"""
    SELECT r.reviewid,
		r.title,
		r.artist,
		r.score,
		g.genre
    FROM reviews AS r
    LEFT JOIN genres as g
	    ON r.reviewid = g.reviewid
    WHERE r.reviewid = '{review_id}'
    """
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    if not result.empty:
        row = result.iloc[0]
        details = {
            "reviewid": row['reviewid'],
            "album": row['title'],
            "score": row['score'],
            "artist": row['artist']
        }
        return details
    else:
        _logs.warning(f'No details found for review ID: {review_id}')
        return {}
    
def get_reviewid_from_custom_id(custom_id:str):
    return custom_id.split('_')[0]

def get_context_data(query:str, collection:chromadb.api.models.Collection, top_n:int):
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )
    context_data = []
    for idx, custom_id in enumerate(results['ids'][0]):
        review_id = get_reviewid_from_custom_id(custom_id)
        details = additional_details(review_id)
        details['text'] = results['documents'][0][idx]
        context_data.append(details)
    return context_data

def get_context(query:str, collection:chromadb.api.models.Collection, top_n:int):
    context_data = get_context_data(query, collection, top_n)
    recommendations = []
    if not context_data:
        return recommendations
    for item in context_data:

        rec = BookReviewData(
            title=item.get('album', 'N/A'),
            artist=item.get('artist', 'N/A'),
            review=item.get('text', 'N/A'),
            score=item.get('score', 0.0)
        )
        recommendations.append(rec)
    return recommendations