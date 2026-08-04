from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# Load environment variables from .env file
load_dotenv() 

# Define constants for the model and database name
MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "data" / "vector_db")

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


print("Done")