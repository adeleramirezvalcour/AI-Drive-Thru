from fastapi import FastAPI
from pydantic import BaseModel

# Bring in previously defined function
from menu import retrieve_menu_context

app = FastAPI()


class QueryRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/menu-search")
def menu_search(request: QueryRequest):

    context = retrieve_menu_context(
        request.text
    )

    return {
        "retrieved_context": context
    }
