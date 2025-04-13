# Example API Requests for Savory Haven Chatbot

This document provides example API requests and responses for testing the Savory Haven restaurant chatbot.

## Chat Endpoint

### Endpoint: `POST /api/chat`

This is the main endpoint for interacting with the chatbot. Below are various example requests and expected responses for different types of inquiries.

## Menu Inquiries

### Example 1: General Menu Inquiry

**Request:**
```json
{
  "user_id": "user123",
  "message": "What's on your menu?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "Our menu features Antipasti, Insalate, Pasta, Pizza, Dolci, and Bevande. We specialize in Italian, Mediterranean, American Fusion cuisine. Would you like to know about any specific category or dietary options?",
  "intent": "menu_inquiry",
  "suggested_actions": ["view_full_menu", "view_specials", "reservation_request"]
}
```

### Example 2: Vegetarian Options

**Request:**
```json
{
  "user_id": "user123",
  "message": "Do you have vegetarian options?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "Yes, we have several vegetarian options including: Bruschetta Classica ($9.50) from our Antipasti, Caprese Salad ($12.95) from our Insalate, Quattro Formaggi Pizza ($18.95) from our Pizza",
  "intent": "menu_inquiry",
  "suggested_actions": ["view_full_menu", "reservation_request"]
}
```

### Example 3: Specific Menu Category

**Request:**
```json
{
  "user_id": "user123",
  "message": "What pasta dishes do you have?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "In our Pasta category, we offer: Pappardelle al Ragu ($22.95)",
  "intent": "menu_inquiry",
  "suggested_actions": ["view_full_menu", "reservation_request"]
}
```

## Reservation Requests

### Example 1: Basic Reservation Inquiry

**Request:**
```json
{
  "user_id": "user123",
  "message": "I'd like to make a reservation",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "I'd be happy to help you with a reservation. Could you please provide the date, time and party size for your reservation?",
  "intent": "reservation_request",
  "suggested_actions": ["call_restaurant", "view_hours"]
}
```

### Example 2: Detailed Reservation Request

**Request:**
```json
{
  "user_id": "user123",
  "message": "I need a table for 4 people on Friday at 7pm",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "I'd be happy to help you book a reservation for 4 people on Friday at 7pm. Please note that this is just a preliminary check. To confirm your reservation, please call us at +1 (555) 987-6543 or visit our website.",
  "intent": "reservation_request",
  "suggested_actions": ["call_restaurant", "visit_website"]
}
```

## Hours and Location Inquiries

### Example 1: General Hours Inquiry

**Request:**
```json
{
  "user_id": "user123",
  "message": "What are your hours?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "Our hours of operation are:\n- Monday: 11:00 to 22:00\n- Tuesday: 11:00 to 22:00\n- Wednesday: 11:00 to 22:00\n- Thursday: 11:00 to 23:00\n- Friday: 11:00 to 00:00\n- Saturday: 10:00 to 00:00\n- Sunday: 10:00 to 21:00\n",
  "intent": "hours_inquiry",
  "suggested_actions": ["location_inquiry", "reservation_request"]
}
```

### Example 2: Specific Day Hours

**Request:**
```json
{
  "user_id": "user123",
  "message": "Are you open on Sundays?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "On Sunday, we're open from 10:00 to 21:00.",
  "intent": "hours_inquiry",
  "suggested_actions": ["view_full_hours", "reservation_request"]
}
```

### Example 3: Location Inquiry

**Request:**
```json
{
  "user_id": "user123",
  "message": "Where are you located?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "We are located at 789 Gourmet Avenue, Flavor Town, CA 90210, USA. Our restaurant is wheelchair accessible. ",
  "intent": "location_inquiry",
  "suggested_actions": ["get_directions", "hours_inquiry"]
}
```

## Other Inquiries

### Example 1: Promotions

**Request:**
```json
{
  "user_id": "user123",
  "message": "Do you have any special offers?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "We currently have 2 special promotions:\n- Summer Wine Pairing: Enjoy a 3-course meal with wine pairings for $65 per person (Available Tuesday-Thursday evenings only. Reservations required.) Available from 2023-06-01 to 2023-08-31.\n- Birthday Special: Get a free dessert when you dine with us on your birthday! (Valid with ID showing birth date. Dine-in only.)\n\nWe also have Happy Hour from 16:00-18:00 on Monday, Tuesday, Wednesday and Thursday featuring:\n- $5 select appetizers\n- $2 off all cocktails\n- Half-price wine by the glass\n",
  "intent": "promotion_inquiry",
  "suggested_actions": ["view_menu", "reservation_request"]
}
```

### Example 2: Contact Information

**Request:**
```json
{
  "user_id": "user123",
  "message": "How can I contact you?",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "You can reach us by phone at +1 (555) 987-6543 or by email at info@savoryhaven.com. You can also visit our website at www.savoryhaven.com. Follow us on Facebook (facebook.com/savoryhaven), Instagram (@savoryhaven) and Twitter (@savoryhaven).",
  "intent": "contact_inquiry",
  "suggested_actions": ["reservation_request", "location_inquiry"]
}
```

### Example 3: Loyalty Program

**Request:**
```json
{
  "user_id": "user123",
  "message": "Tell me about your loyalty program",
  "session_id": "session456"
}
```

**Expected Response:**
```json
{
  "response": "Our loyalty program, Savory Circle, offers the following benefits:\n- Earn 1 point for every $1 spent\n- Free appetizer after 100 points\n- Birthday reward\n- Exclusive event invitations\n\nWe have different membership tiers:\n- Silver (requires 0 points): Basic rewards\n- Gold (requires 500 points): Priority reservations, Complimentary dessert monthly\n",
  "intent": "loyalty_program",
  "suggested_actions": ["join_loyalty", "reservation_request"]
}
```

## Conversation History Endpoints

### Endpoint: `GET /api/conversations/{user_id}`

**Example Request:**
```
GET /api/conversations/user123
```

**Expected Response:**
```json
[
  {
    "id": 1,
    "user_id": "user123",
    "session_id": "session456",
    "user_message": "What's on your menu?",
    "detected_intent": "menu_inquiry",
    "entities": {},
    "bot_response": "Our menu features Antipasti, Insalate, Pasta, Pizza, Dolci, and Bevande. We specialize in Italian, Mediterranean, American Fusion cuisine. Would you like to know about any specific category or dietary options?",
    "suggested_actions": ["view_full_menu", "view_specials", "reservation_request"],
    "timestamp": "2025-04-12T19:05:00.000Z"
  },
  {
    "id": 2,
    "user_id": "user123",
    "session_id": "session456",
    "user_message": "Do you have vegetarian options?",
    "detected_intent": "menu_inquiry",
    "entities": {
      "dietary_preference": ["vegetarian"]
    },
    "bot_response": "Yes, we have several vegetarian options including: Bruschetta Classica ($9.50) from our Antipasti, Caprese Salad ($12.95) from our Insalate, Quattro Formaggi Pizza ($18.95) from our Pizza",
    "suggested_actions": ["view_full_menu", "reservation_request"],
    "timestamp": "2025-04-12T19:06:00.000Z"
  }
]
```

### Endpoint: `GET /api/conversations/session/{session_id}`

**Example Request:**
```
GET /api/conversations/session/session456
```

**Expected Response:**
```json
[
  {
    "id": 1,
    "user_id": "user123",
    "session_id": "session456",
    "user_message": "What's on your menu?",
    "detected_intent": "menu_inquiry",
    "entities": {},
    "bot_response": "Our menu features Antipasti, Insalate, Pasta, Pizza, Dolci, and Bevande. We specialize in Italian, Mediterranean, American Fusion cuisine. Would you like to know about any specific category or dietary options?",
    "suggested_actions": ["view_full_menu", "view_specials", "reservation_request"],
    "timestamp": "2025-04-12T19:05:00.000Z"
  },
  {
    "id": 2,
    "user_id": "user123",
    "session_id": "session456",
    "user_message": "Do you have vegetarian options?",
    "detected_intent": "menu_inquiry",
    "entities": {
      "dietary_preference": ["vegetarian"]
    },
    "bot_response": "Yes, we have several vegetarian options including: Bruschetta Classica ($9.50) from our Antipasti, Caprese Salad ($12.95) from our Insalate, Quattro Formaggi Pizza ($18.95) from our Pizza",
    "suggested_actions": ["view_full_menu", "reservation_request"],
    "timestamp": "2025-04-12T19:06:00.000Z"
  }
]
```
