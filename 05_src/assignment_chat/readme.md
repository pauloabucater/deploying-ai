# Assignment 2 - Books and Music chat

The goal with the assignment was to build a conversational interface that would use agents to get information about books and music.

## Services

The services were implemented using LangGraph. The chat is associated with an LLM model and tools in the main.py file. Each tools was stored in separate files: tools_books.py and tools_music.py.

### Service 1 - Music API

This tool calls an API from MusicBrainz.org, which provides a catalog search based on different entities: annotation, area, artist, cdstub, event, instrument, label, place, recording, release, release-group, series, tag, work, url. 

### Service 2 - Books query (embeddings)

A dataset from Kaggle with 10K Amazon books metadata was downloaded and used as file db (data/books_data.csv). Embeddings were produced and stored into a local ChromaDB folder (data/chromadb).

The embeddings were created using the Jupiter notebook embeddings_books.ipynb.

Using similarity search, the keys were retrieved from ChromaDB and the response augmented with other metadata from the csv file through pandas.

### Service 3

Not implemented

## User Interface

Implemented in Gradio. The conversational tone defined in the system prompt was GenZ style.

## Guardrails

Instructions for guardrails were added to the system prompt file (prompts.py)
