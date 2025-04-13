from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import uuid

from app.database.db import get_db
from app.database.models import ConversationLog
from app.database.crud import log_conversation
from app.nlp.intent_entity import IntentClassifier, EntityExtractor
from app.nlp.response_generator import ResponseGenerator

router = APIRouter()

# Initialize NLP components
intent_classifier = IntentClassifier()
entity_extractor = EntityExtractor()

@router.post("/chat")
async def chat(
    request: Request,
    chat_request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Process a chat message and return a response.
    
    Request body:
    {
        "user_id": "string",
        "message": "string",
        "session_id": "string" (optional)
    }
    
    Response:
    {
        "response": "string",
        "intent": "string",
        "suggested_actions": ["string"]
    }
    """
    # Get restaurant data from app state
    restaurant_data = request.app.state.restaurant_data
    
    # Initialize response generator with restaurant data
    response_generator = ResponseGenerator(restaurant_data)
    
    # Extract request data
    user_id = chat_request.get("user_id", "anonymous")
    user_message = chat_request.get("message", "")
    session_id = chat_request.get("session_id", str(uuid.uuid4()))
    
    # Validate request
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Classify intent
    intent, confidence = intent_classifier.classify(user_message)
    
    # Extract entities
    entities = entity_extractor.extract_entities(user_message)
    
    # Generate response
    response_text, suggested_actions = response_generator.generate_response(intent, entities)
    
    # Log conversation
    log_entry = log_conversation(
        db=db,
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        detected_intent=intent,
        entities=entities,
        bot_response=response_text,
        suggested_actions=suggested_actions
    )
    
    # Return response
    return {
        "response": response_text,
        "intent": intent,
        "suggested_actions": suggested_actions
    }

@router.get("/conversations/{user_id}")
async def get_user_conversations(
    user_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get conversation history for a specific user.
    """
    from app.database.crud import get_conversation_history
    
    conversations = get_conversation_history(db, user_id=user_id, limit=limit)
    return [conv.to_dict() for conv in conversations]

@router.get("/conversations/session/{session_id}")
async def get_session_conversations(
    session_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get conversation history for a specific session.
    """
    from app.database.crud import get_conversation_history
    
    conversations = get_conversation_history(db, session_id=session_id, limit=limit)
    return [conv.to_dict() for conv in conversations]
