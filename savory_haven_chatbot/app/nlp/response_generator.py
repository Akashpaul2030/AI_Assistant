from typing import Dict, List, Optional, Any, Tuple
import json
import random
from pathlib import Path
from app.nlp.intent_entity import IntentClassifier, EntityExtractor

class ResponseGenerator:
    """
    Generates appropriate responses based on detected intent and entities.
    Uses the restaurant dataset to provide relevant information.
    """
    
    def __init__(self, restaurant_data: Dict[str, Any]):
        """
        Initialize the response generator with restaurant data.
        
        Args:
            restaurant_data: Dictionary containing restaurant information
        """
        self.restaurant_data = restaurant_data
        
    def generate_response(self, intent: str, entities: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Generate a response based on the detected intent and entities.
        
        Args:
            intent: Detected intent from the user message
            entities: Extracted entities from the user message
            
        Returns:
            Tuple of (response_text, suggested_actions)
        """
        if intent == "greeting":
            return self._handle_greeting(), ["menu_inquiry", "reservation_request", "hours_inquiry"]
        
        elif intent == "farewell":
            return self._handle_farewell(), []
        
        elif intent == "menu_inquiry":
            return self._handle_menu_inquiry(entities)
        
        elif intent == "reservation_request":
            return self._handle_reservation_request(entities)
        
        elif intent == "hours_inquiry":
            return self._handle_hours_inquiry(entities)
        
        elif intent == "location_inquiry":
            return self._handle_location_inquiry(), ["get_directions", "hours_inquiry"]
        
        elif intent == "contact_inquiry":
            return self._handle_contact_inquiry(), ["reservation_request", "send_email"]
        
        elif intent == "promotion_inquiry":
            return self._handle_promotion_inquiry(), ["view_promotions", "reservation_request"]
        
        elif intent == "loyalty_program":
            return self._handle_loyalty_program(), ["join_loyalty", "view_benefits"]
        
        elif intent == "general_faq":
            return self._handle_general_faq(entities)
        
        else:
            return self._handle_unknown(), ["menu_inquiry", "reservation_request", "contact_inquiry"]
    
    def _handle_greeting(self) -> str:
        """Handle greeting intent."""
        greetings = [
            f"Welcome to {self.restaurant_data['restaurant']['name']}! How can I assist you today?",
            f"Hello! Thanks for contacting {self.restaurant_data['restaurant']['name']}. What can I help you with?",
            f"Hi there! I'm the virtual assistant for {self.restaurant_data['restaurant']['name']}. How may I help you?"
        ]
        return random.choice(greetings)
    
    def _handle_farewell(self) -> str:
        """Handle farewell intent."""
        farewells = [
            f"Thank you for chatting with {self.restaurant_data['restaurant']['name']}. Have a great day!",
            "It was a pleasure assisting you. Hope to see you soon!",
            "Thank you for your interest in our restaurant. Have a wonderful day!"
        ]
        return random.choice(farewells)
    
    def _handle_menu_inquiry(self, entities: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Handle menu inquiry intent."""
        # Check for dietary preferences
        if 'dietary_preference' in entities:
            return self._handle_dietary_menu_inquiry(entities['dietary_preference']), ["view_full_menu", "reservation_request"]
        
        # Check for specific menu category
        if 'menu_category' in entities:
            return self._handle_category_menu_inquiry(entities['menu_category']), ["view_full_menu", "reservation_request"]
        
        # General menu inquiry
        menu_categories = [category["name"] for category in self.restaurant_data["menu"]["categories"]]
        categories_text = ", ".join(menu_categories[:-1]) + ", and " + menu_categories[-1] if len(menu_categories) > 1 else menu_categories[0]
        
        response = f"Our menu features {categories_text}. We specialize in {', '.join(self.restaurant_data['restaurant']['cuisine_types'])} cuisine. Would you like to know about any specific category or dietary options?"
        
        return response, ["view_full_menu", "view_specials", "reservation_request"]
    
    def _handle_dietary_menu_inquiry(self, preferences: List[str]) -> str:
        """Handle menu inquiry with dietary preferences."""
        preference = preferences[0].lower()
        
        # Find menu items matching the dietary preference
        matching_items = []
        
        for category in self.restaurant_data["menu"]["categories"]:
            for item in category["items"]:
                if preference == "vegetarian" and item.get("vegetarian", False):
                    matching_items.append((item["name"], category["name"], item["price"]))
                elif preference == "vegan" and item.get("vegan", False):
                    matching_items.append((item["name"], category["name"], item["price"]))
                elif preference == "gluten-free" and item.get("gluten_free", False):
                    matching_items.append((item["name"], category["name"], item["price"]))
        
        if matching_items:
            response = f"Yes, we have several {preference} options including: "
            items_text = [f"{name} (${price:.2f}) from our {category}" for name, category, price in matching_items[:3]]
            response += ", ".join(items_text)
            
            if len(matching_items) > 3:
                response += f", and {len(matching_items) - 3} more items"
            
            return response
        else:
            return f"We have limited {preference} options on our regular menu, but our chef can accommodate special dietary needs. Please let us know when making a reservation."
    
    def _handle_category_menu_inquiry(self, categories: List[str]) -> str:
        """Handle menu inquiry for specific category."""
        category = categories[0].lower()
        
        # Map common terms to menu categories
        category_map = {
            "appetizer": "Antipasti",
            "starter": "Antipasti",
            "salad": "Insalate",
            "pasta": "Pasta",
            "pizza": "Pizza",
            "dessert": "Dolci",
            "drink": "Bevande",
            "beverage": "Bevande",
            "wine": "Bevande",
            "cocktail": "Bevande"
        }
        
        mapped_category = category_map.get(category, category.capitalize())
        
        # Find the matching category in the menu
        for menu_category in self.restaurant_data["menu"]["categories"]:
            if menu_category["name"].lower() == mapped_category.lower():
                items_text = [f"{item['name']} (${item['price']:.2f})" for item in menu_category["items"][:5]]
                
                response = f"In our {menu_category['name']} category, we offer: {', '.join(items_text)}"
                
                if len(menu_category["items"]) > 5:
                    response += f", and {len(menu_category['items']) - 5} more items"
                
                return response
        
        return f"I don't have specific information about {category} items, but our full menu includes Antipasti, Insalate, Pasta, Pizza, Dolci, and Bevande. Would you like to know about any of these categories?"
    
    def _handle_reservation_request(self, entities: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Handle reservation request intent."""
        # Check if we have date, time, and party size
        has_date = 'date' in entities
        has_time = 'time' in entities
        has_party_size = 'party_size' in entities
        
        if has_date and has_time and has_party_size:
            return f"I'd be happy to help you book a reservation for {entities['party_size']} people on {entities['date']} at {entities['time']}. Please note that this is just a preliminary check. To confirm your reservation, please call us at {self.restaurant_data['restaurant']['contact']['phone']} or visit our website.", ["call_restaurant", "visit_website"]
        
        # Missing some information
        missing_info = []
        if not has_date:
            missing_info.append("date")
        if not has_time:
            missing_info.append("time")
        if not has_party_size:
            missing_info.append("party size")
        
        if missing_info:
            missing_text = ", ".join(missing_info[:-1]) + (" and " if len(missing_info) > 1 else "") + missing_info[-1] if len(missing_info) > 1 else missing_info[0]
            return f"I'd be happy to help you with a reservation. Could you please provide the {missing_text} for your reservation?", ["call_restaurant", "view_hours"]
        
        return "I'd be happy to help you make a reservation. You can book a table by calling us at " + self.restaurant_data['restaurant']['contact']['phone'] + " or through our website.", ["call_restaurant", "visit_website"]
    
    def _handle_hours_inquiry(self, entities: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Handle hours inquiry intent."""
        # Check if asking about specific day
        if 'day_of_week' in entities:
            day = entities['day_of_week'][0].lower()
            
            if day in self.restaurant_data['restaurant']['hours']:
                hours = self.restaurant_data['restaurant']['hours'][day]
                return f"On {day.capitalize()}, we're open from {hours['open']} to {hours['close']}.", ["view_full_hours", "reservation_request"]
            
            elif day == "weekday":
                response = "Our weekday hours are:\n"
                for weekday in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                    hours = self.restaurant_data['restaurant']['hours'][weekday]
                    response += f"- {weekday.capitalize()}: {hours['open']} to {hours['close']}\n"
                return response, ["view_weekend_hours", "reservation_request"]
            
            elif day == "weekend":
                response = "Our weekend hours are:\n"
                for weekend_day in ["saturday", "sunday"]:
                    hours = self.restaurant_data['restaurant']['hours'][weekend_day]
                    response += f"- {weekend_day.capitalize()}: {hours['open']} to {hours['close']}\n"
                return response, ["view_weekday_hours", "reservation_request"]
        
        # General hours inquiry
        response = "Our hours of operation are:\n"
        for day, hours in self.restaurant_data['restaurant']['hours'].items():
            response += f"- {day.capitalize()}: {hours['open']} to {hours['close']}\n"
        
        return response, ["location_inquiry", "reservation_request"]
    
    def _handle_location_inquiry(self) -> Tuple[str, List[str]]:
        """Handle location inquiry intent."""
        address = self.restaurant_data['restaurant']['address']
        formatted_address = f"{address['street']}, {address['city']}, {address['state']} {address['zip']}, {address['country']}"
        
        response = f"We are located at {formatted_address}. "
        
        if "features" in self.restaurant_data['restaurant']:
            if "Wheelchair Accessible" in self.restaurant_data['restaurant']['features']:
                response += "Our restaurant is wheelchair accessible. "
            
            if "Parking Available" in self.restaurant_data['restaurant']['features']:
                response += "Parking is available on-site. "
        
        return response, ["get_directions", "hours_inquiry"]
    
    def _handle_contact_inquiry(self) -> Tuple[str, List[str]]:
        """Handle contact inquiry intent."""
        contact = self.restaurant_data['restaurant']['contact']
        response = f"You can reach us by phone at {contact['phone']} or by email at {contact['email']}. "
        
        if 'website' in contact:
            response += f"You can also visit our website at {contact['website']}. "
        
        if 'social_media' in self.restaurant_data['restaurant']:
            social = self.restaurant_data['restaurant']['social_media']
            platforms = []
            
            if 'facebook' in social:
                platforms.append(f"Facebook ({social['facebook']})")
            if 'instagram' in social:
                platforms.append(f"Instagram ({social['instagram']})")
            if 'twitter' in social:
                platforms.append(f"Twitter ({social['twitter']})")
            
            if platforms:
                platforms_text = ", ".join(platforms[:-1]) + (" and " if len(platforms) > 1 else "") + platforms[-1] if len(platforms) > 1 else platforms[0]
                response += f"Follow us on {platforms_text}."
        
        return response, ["reservation_request", "location_inquiry"]
    
    def _handle_promotion_inquiry(self) -> Tuple[str, List[str]]:
        """Handle promotion inquiry intent."""
        if 'promotions' in self.restaurant_data and self.restaurant_data['promotions']:
            promotions = self.restaurant_data['promotions']
            response = f"We currently have {len(promotions)} special promotions:\n"
            
            for promo in promotions:
                response += f"- {promo['name']}: {promo['description']}"
                
                if 'terms' in promo:
                    response += f" ({promo['terms']})"
                
                if 'valid_from' in promo and 'valid_to' in promo:
                    response += f" Available from {promo['valid_from']} to {promo['valid_to']}."
                
                response += "\n"
            
            # Check for happy hour specials
            if 'specials' in self.restaurant_data['menu'] and 'happy_hour' in self.restaurant_data['menu']['specials']:
                happy_hour = self.restaurant_data['menu']['specials']['happy_hour']
                days = [day.capitalize() for day in happy_hour['days']]
                days_text = ", ".join(days[:-1]) + (" and " if len(days) > 1 else "") + days[-1] if len(days) > 1 else days[0]
                
                response += f"\nWe also have Happy Hour from {happy_hour['times']} on {days_text} featuring:\n"
                for offer in happy_hour['offers']:
                    response += f"- {offer}\n"
        else:
            response = "We don't have any current promotions listed, but please check our website or social media for the latest offers."
        
        return response, ["view_menu", "reservation_request"]
    
    def _handle_loyalty_program(self) -> Tuple[str, List[str]]:
        """Handle loyalty program inquiry intent."""
        if 'loyalty_program' in self.restaurant_data and self.restaurant_data['loyalty_program']:
            loyalty = self.restaurant_data['loyalty_program']
            
            response = f"Our loyalty program, {loyalty['name']}, offers the following benefits:\n"
            
            for benefit in loyalty['benefits']:
                response += f"- {benefit}\n"
            
            if 'tiers' in loyalty:
                response += "\nWe have different membership tiers:\n"
                
                for tier in loyalty['tiers']:
                    response += f"- {tier['name']} (requires {tier['points_required']} points): {', '.join(tier['benefits'])}\n"
        else:
            response = "I don't have information about our loyalty program. Please ask our staff during your next visit or check our website for details."
        
        return response, ["join_loyalty", "reservation_request"]
    
    def _handle_general_faq(self, entities: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Handle general FAQ intent."""
        # Try to find matching FAQ
        if 'faqs' in self.restaurant_data:
            for category in self.restaurant_data['faqs']:
                for qa in category['questions']:
                    # Simple keyword matching
                    question_lower = qa['question'].lower()
                    
                    # Check for dietary keywords in question
                    if 'dietary_preference' in entities:
                        for pref in entities['dietary_preference']:
                            if pref.lower() in question_lower:
                                return qa['answer'], ["menu_inquiry", "reservation_request"]
                    
                    # Check for menu keywords in question
                    if 'menu_category' in entities:
                        for cat in entities['menu_category']:
                            if cat.lower() in question_lower:
                                return qa['answer'], ["menu_inquiry", "view_full_menu"]
                    
                    # Check for reservation keywords
                    if 'reservation' in question_lower and 'reservation_request' in entities.get('intent_history', []):
                        return qa['answer'], ["make_reservation", "contact_inquiry"]
        
        # If no matching FAQ found
        return "I'm not sure I understand your question. Could you please rephrase or ask about our menu, hours, location, or making a reservation?", ["menu_inquiry", "hours_inquiry", "location_inquiry", "reservation_request"]
    
    def _handle_unknown(self) -> Tuple[str, List[str]]:
        """Handle unknown intent."""
        responses = [
            "I'm not sure I understand what you're asking. Could you please rephrase or ask about our menu, hours, location, or making a reservation?",
            "I didn't quite catch that. How can I help you with our restaurant services?",
            f"I'm the virtual assistant for {self.restaurant_data['restaurant']['name']}. I can help with menu information, reservations, hours, and location. What would you like to know?"
        ]
        
        return random.choice(responses), ["menu_inquiry", "hours_inquiry", "location_inquiry", "reservation_request"]
