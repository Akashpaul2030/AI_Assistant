from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import os
from pathlib import Path

from app.database.db import get_db, init_db
from app.endpoints.chat import router as chat_router

# Create FastAPI app
app = FastAPI(
    title="Savory Haven Restaurant Chatbot API",
    description="A conversational AI system for Savory Haven restaurant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api")

# Load restaurant data
@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_db()
    
    # Load restaurant data
    app.state.restaurant_data = {}
    data_path = Path(__file__).parent / "res-bot-dataset.json"
    
    if data_path.exists():
        with open(data_path, "r") as f:
            app.state.restaurant_data = json.load(f)
    else:
        print(f"Warning: Restaurant data file not found at {data_path}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint that returns a simple HTML page with instructions.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Savory Haven Restaurant Chatbot API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #4a4a4a;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            h2 {
                color: #5a5a5a;
                margin-top: 30px;
            }
            pre {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            code {
                font-family: monospace;
            }
            .endpoint {
                margin-bottom: 30px;
            }
            .method {
                font-weight: bold;
                color: #0066cc;
            }
        </style>
    </head>
    <body>
        <h1>Welcome to Savory Haven Restaurant Chatbot API</h1>
        
        <p>This API provides a conversational interface for the Savory Haven restaurant, allowing users to inquire about the menu, make reservations, check hours, and more.</p>
        
        <h2>API Documentation</h2>
        <p>Visit the <a href="/docs">Swagger UI documentation</a> to explore and test the API endpoints.</p>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <p><span class="method">POST</span> /api/chat</p>
            <p>Send a message to the chatbot and receive a response.</p>
            <p>Example request:</p>
            <pre><code>
{
  "user_id": "user123",
  "message": "Do you have vegetarian options?",
  "session_id": "session456"
}
            </code></pre>
            <p>Example response:</p>
            <pre><code>
{
  "response": "Yes, we have several vegetarian options including Bruschetta Classica, Caprese Salad, and Quattro Formaggi Pizza.",
  "intent": "menu_inquiry",
  "suggested_actions": ["view_vegetarian_menu", "make_reservation"]
}
            </code></pre>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> /api/conversations/{user_id}</p>
            <p>Retrieve conversation history for a specific user.</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> /api/conversations/session/{session_id}</p>
            <p>Retrieve conversation history for a specific session.</p>
        </div>
        
        <h2>Running the API</h2>
        <p>To run the API locally:</p>
        <pre><code>uvicorn app.main:app --reload</code></pre>
    </body>
    </html>
    """
    return html_content

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Savory Haven Restaurant Chatbot API",
        version="1.0.0",
        description="A conversational AI system for Savory Haven restaurant",
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
