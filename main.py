# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello, FastAPI!"}


import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from langchain.chat_models import ChatGoogleGemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Langchain components
# llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(
    google_api_key=api_key,
    model="gemini-2.5-flash"
)
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""You are a travel planner agent. Your task is to provide travel recommendations based on the user's input.
    The user will provide a country or city name.
    Return a list of top destinations or places where visitors can visit in that country or city.
    For each destination, provide a brief description and why it's a good place to visit.
    Provide a concise and informative response, formatted as a list.

    User input: {user_input}
    """
)

# Refactor LLMChain to use RunnableSequence and StrOutputParser
llm_chain = prompt | llm | StrOutputParser()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "travel_recommendations": None})

@app.post("/plan_travel")
async def plan_travel(request: Request, user_input: str = Form(...)):
    try:
        print(f"Received input for travel planning: {user_input}")
        travel_recommendations = llm_chain.invoke({"user_input": user_input})
        print(f"Travel recommendations: {travel_recommendations}")
        return {"response": travel_recommendations}
    except Exception as e:
        print(f"Error in plan_travel: {e}")
        return {"response": f"Error: {e}"}
