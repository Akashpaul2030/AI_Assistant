import re
import json
from typing import Dict, List, Tuple, Optional, Any
import random

# Define intent categories
INTENT_CATEGORIES = {
    "menu_inquiry": [
        "menu", "food", "dish", "eat", "cuisine", "serve", "order", 
        "vegetarian", "vegan", "gluten", "allerg", "special diet"
    ],
    "reservation_request": [
        "book", "reserve", "reservation", "table", "seat", "party", 
        "people", "person", "group", "available", "availability"
    ],
    "hours_inquiry": [
        "hour", "open", "close", "time", "when", "day", "schedule", "operation"
    ],
    "location_inquiry": [
        "where", "location", "address", "direction", "find", "map", "located"
    ],
    "contact_inquiry": [
        "contact", "phone", "call", "email", "reach", "number"
    ],
    "promotion_inquiry": [
        "deal", "special", "offer", "discount", "promotion", "coupon", "happy hour"
    ],
    "loyalty_program": [
        "loyalty", "reward", "point", "member", "program", "savory circle"
    ],
    "general_faq": [
        "question", "faq", "help", "info", "information", "tell me about"
    ],
    "greeting": [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"
    ],
    "farewell": [
        "bye", "goodbye", "see you", "farewell", "thanks", "thank you"
    ]
}

class IntentClassifier:
    """
    A simple rule-based intent classifier for restaurant chatbot.
    """
    
    def __init__(self, intent_categories: Dict[str, List[str]] = None):
        """
        Initialize the intent classifier with predefined intent categories.
        
        Args:
            intent_categories: Dictionary mapping intent names to lists of keywords
        """
        self.intent_categories = intent_categories or INTENT_CATEGORIES
    
    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify the user message into an intent category.
        
        Args:
            message: User message text
            
        Returns:
            Tuple of (intent_name, confidence_score)
        """
        message = message.lower()
        
        # Initialize scores for each intent
        scores = {intent: 0 for intent in self.intent_categories}
        
        # Calculate score for each intent based on keyword matches
        for intent, keywords in self.intent_categories.items():
            for keyword in keywords:
                if keyword.lower() in message:
                    scores[intent] += 1
        
        # Find the intent with the highest score
        max_score = max(scores.values()) if scores else 0
        
        if max_score == 0:
            return "unknown", 0.0
        
        # Get all intents with the max score
        top_intents = [intent for intent, score in scores.items() if score == max_score]
        
        # If there's a tie, prioritize certain intents
        priority_order = [
            "reservation_request",
            "menu_inquiry",
            "hours_inquiry",
            "location_inquiry",
            "promotion_inquiry",
            "loyalty_program",
            "contact_inquiry",
            "general_faq",
            "greeting",
            "farewell"
        ]
        
        for priority_intent in priority_order:
            if priority_intent in top_intents:
                # Calculate confidence based on number of keyword matches and message length
                confidence = min(1.0, max_score / (len(message.split()) * 0.5))
                return priority_intent, confidence
        
        # If no priority match, return the first top intent
        confidence = min(1.0, max_score / (len(message.split()) * 0.5))
        return top_intents[0], confidence


class EntityExtractor:
    """
    A simple rule-based entity extractor for restaurant chatbot.
    """
    
    def __init__(self):
        """
        Initialize the entity extractor with regex patterns for common entities.
        """
        # Date patterns (MM/DD/YYYY, MM-DD-YYYY, Month DD, etc.)
        self.date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b'
        
        # Time patterns (HH:MM AM/PM, H:MM AM/PM, etc.)
        self.time_pattern = r'\b(?:\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?|\d{1,2}\s*(?:am|pm|AM|PM))\b'
        
        # Party size patterns (X people, party of X, group of X, etc.)
        self.party_size_pattern = r'\b(?:(?:party|group|table|reservation)\s+(?:of|for)\s+(\d+)|(\d+)\s+(?:people|persons|guests|diners))\b'
        
        # Dietary preference patterns
        self.dietary_pattern = r'\b(vegetarian|vegan|gluten[- ]free|dairy[- ]free|nut[- ]free|pescatarian|halal|kosher)\b'
        
        # Menu item or category patterns
        self.menu_pattern = r'\b(appetizers?|starters?|salads?|pasta|pizza|main course|entrees?|desserts?|drinks?|beverages?|wines?|cocktails?)\b'
        
        # Day of week patterns
        self.day_pattern = r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend)\b'
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities from the user message.
        
        Args:
            message: User message text
            
        Returns:
            Dictionary of extracted entities
        """
        message = message.lower()
        entities = {}
        
        # Extract dates
        date_matches = re.findall(self.date_pattern, message, re.IGNORECASE)
        if date_matches:
            entities['date'] = date_matches[0]
        
        # Extract times
        time_matches = re.findall(self.time_pattern, message, re.IGNORECASE)
        if time_matches:
            entities['time'] = time_matches[0]
        
        # Extract party size
        party_size_matches = re.findall(self.party_size_pattern, message, re.IGNORECASE)
        if party_size_matches:
            # Handle the two capture groups in the regex
            for match in party_size_matches:
                if match[0]:  # First capture group (party of X)
                    entities['party_size'] = int(match[0])
                    break
                elif match[1]:  # Second capture group (X people)
                    entities['party_size'] = int(match[1])
                    break
        
        # Extract dietary preferences
        dietary_matches = re.findall(self.dietary_pattern, message, re.IGNORECASE)
        if dietary_matches:
            entities['dietary_preference'] = dietary_matches
        
        # Extract menu items or categories
        menu_matches = re.findall(self.menu_pattern, message, re.IGNORECASE)
        if menu_matches:
            entities['menu_category'] = menu_matches
        
        # Extract days of the week
        day_matches = re.findall(self.day_pattern, message, re.IGNORECASE)
        if day_matches:
            entities['day_of_week'] = day_matches
        
        return entities
