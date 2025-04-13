# Savory Haven Restaurant Chatbot

A conversational AI system for Savory Haven restaurant built with Python and FastAPI.

## Overview

This application provides a chatbot API for Savory Haven restaurant that can:

- Parse user messages to determine their intent (menu inquiry, reservation request, etc.)
- Extract relevant entities (date, time, party size, dietary preferences, etc.)
- Respond with appropriate information from the restaurant dataset
- Log each interaction to a SQLite database for analytics
- Return structured JSON responses with suggested actions

## Project Structure

```
savory_haven_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── res-bot-dataset.json
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── models.py
│   │   └── crud.py
│   ├── endpoints/
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── intent_entity.py
│   │   └── response_generator.py
│   └── utils/
│       └── __init__.py
├── requirements.txt
├── TESTING.md
└── EXAMPLES.md
```

## Features

- **Intent Classification**: Identifies user intent from messages (menu inquiry, reservation request, etc.)
- **Entity Extraction**: Extracts relevant information like dates, times, party sizes, and dietary preferences
- **Response Generation**: Creates contextual responses based on the restaurant dataset
- **Conversation Logging**: Stores all interactions in a SQLite database
- **API Endpoints**: Provides RESTful endpoints for chat interaction and conversation history
- **Documentation**: Includes Swagger UI for API testing and exploration

## Technical Implementation

### Database

- SQLite database with SQLAlchemy ORM
- `ConversationLog` model for storing user interactions
- CRUD operations for logging and retrieving conversations

### NLP Components

- Rule-based intent classification with confidence scoring
- Regex-based entity extraction for dates, times, party sizes, etc.
- Context-aware response generation using the restaurant dataset

### API Endpoints

- `POST /api/chat`: Main endpoint for chatbot interaction
- `GET /api/conversations/{user_id}`: Retrieve conversation history for a user
- `GET /api/conversations/session/{session_id}`: Retrieve conversation history for a session

## Getting Started

See [TESTING.md](TESTING.md) for detailed instructions on running and testing the application.

## Example Requests and Responses

See [EXAMPLES.md](EXAMPLES.md) for example API requests and expected responses.

## Requirements

- Python 3.11+
- FastAPI
- SQLAlchemy
- Uvicorn
- Other dependencies listed in requirements.txt

## Future Enhancements

- Implement more sophisticated NLP using spaCy or a small transformer model
- Add authentication for API access
- Implement a web interface for direct user interaction
- Add support for more complex conversation flows
- Integrate with external reservation systems
