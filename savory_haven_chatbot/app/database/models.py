from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.sql import func
from app.database.db import Base

class ConversationLog(Base):
    """
    Model for storing conversation logs between users and the chatbot.
    """
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    session_id = Column(String(50), index=True)
    user_message = Column(Text, nullable=False)
    detected_intent = Column(String(100))
    entities = Column(JSON, nullable=True)
    bot_response = Column(Text, nullable=False)
    suggested_actions = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "detected_intent": self.detected_intent,
            "entities": self.entities,
            "bot_response": self.bot_response,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
