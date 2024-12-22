#!/usr/bin/env python3

# Load packages.
import os
import chromadb
import chromadb.config

from chromadb.server.fastapi import FastAPI


# Get the port from the environment.
CHROMA_PORT = os.environ.get("CHROMA_PORT", "8878")

# Configure settings.
settings = chromadb.config.Settings()

# Instantiate the app.
server = FastAPI(settings)
app = server.app()


@app.get("/", include_in_schema=False)
def read_root():
    """
    Simple landing page.
    """
    
    documentation_page = f"http://localhost:{CHROMA_PORT}{app.docs_url}"
    message = {"message": f"Please got to {documentation_page} to see the API documentation."}

    return message