# Running and Testing the Savory Haven Chatbot

This document provides instructions for running and testing the Savory Haven restaurant chatbot application.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation

1. Clone or download the project repository
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application locally, use the following command from the project root directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This will start the FastAPI server with hot reloading enabled, making the API accessible at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the automatically generated Swagger UI documentation at:

```
http://localhost:8000/docs
```

This interactive documentation allows you to:
- View all available endpoints
- Test the API directly from the browser
- See request and response schemas

## Testing the Chatbot

### Using Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Find the `/api/chat` POST endpoint
3. Click "Try it out"
4. Enter a request body in the following format:

```json
{
  "user_id": "test_user",
  "message": "Do you have vegetarian options?",
  "session_id": "test_session"
}
```

5. Click "Execute" to send the request
6. View the response

### Using curl

You can also test the API using curl from the command line:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "test_user",
  "message": "Do you have vegetarian options?",
  "session_id": "test_session"
}'
```

### Using Postman

1. Open Postman
2. Create a new POST request to `http://localhost:8000/api/chat`
3. Set the request header `Content-Type` to `application/json`
4. In the request body, select "raw" and "JSON", then enter:

```json
{
  "user_id": "test_user",
  "message": "Do you have vegetarian options?",
  "session_id": "test_session"
}
```

5. Click "Send" to execute the request
6. View the response

## Example Queries

Here are some example queries you can use to test different intents:

### Menu Inquiries

- "What's on your menu?"
- "Do you have vegetarian options?"
- "What pasta dishes do you serve?"
- "Do you have gluten-free options?"
- "What are your most popular dishes?"

### Reservation Requests

- "I'd like to make a reservation"
- "Can I book a table for 4 people on Friday at 7pm?"
- "Do you have availability for 6 people tomorrow night?"
- "I need a table with wheelchair access"

### Hours and Location

- "What are your hours?"
- "Are you open on Sundays?"
- "What time do you close on Friday?"
- "Where are you located?"
- "What's your address?"

### Other Inquiries

- "Do you have happy hour specials?"
- "Tell me about your loyalty program"
- "How can I contact the restaurant?"
- "Do you offer catering?"

## Viewing Conversation Logs

You can view conversation logs for a specific user or session using the following endpoints:

- Get user conversations: `GET /api/conversations/{user_id}`
- Get session conversations: `GET /api/conversations/session/{session_id}`

For example:

```
http://localhost:8000/api/conversations/test_user
```

## Database

The application uses SQLite for simplicity. The database file will be created automatically in the project root directory when the application starts.

To view the database contents directly, you can use any SQLite browser or the SQLite command line:

```bash
sqlite3 savory_haven_chatbot.db
```

Then, to view conversation logs:

```sql
SELECT * FROM conversation_logs;
```
