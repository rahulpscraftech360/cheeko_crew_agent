from flask import Flask, render_template, request, jsonify
import json
import time
from crewai import Agent, Task, Crew, LLM
import re
from datetime import datetime
import pytz

app = Flask(__name__)
def datetimeformat(value):
    try:
        # Convert Unix timestamp to IST datetime
        ist = pytz.timezone('Asia/Kolkata')
        dt = datetime.fromtimestamp(float(value), tz=ist)
        return dt.strftime('%H:%M')  # Format as HH:MM
    except (ValueError, TypeError):
        return "Unknown time"

app.jinja_env.filters['datetimeformat'] = datetimeformat
# Initialize LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=0.9,
    stop=None,
    stream=False,
)

# Session and user tracking
session_start_time = time.time()
interaction_count = 0
user_profile = {"name": "", "age": 7, "favorite_theme": ""}
points = 0
parental_log = []
chat_history = []
MAX_HISTORY = 20

# Define agents with Cheeko branding
root_kids_agent = Agent(
    role="Central Decision Maker",
    goal="Route tasks to agents for safe, fun chats",
    backstory="Cheeko's brain, picking the best buddy for kids' chats.",
    llm=llm,
    verbose=True
)

inappropriate_filter_agent = Agent(
    role="Content Safety Guardian",
    goal="Keep chats safe for kids",
    backstory="Cheeko's protector, ensuring kid-friendly messages.",
    llm=llm,
    verbose=True
)

overuse_monitor_agent = Agent(
    role="Screen Time Manager",
    goal="Encourage healthy chat habits",
    backstory="Cheeko's guide, promoting breaks for kids.",
    llm=llm,
    verbose=True
)

help_escalation_agent = Agent(
    role="Parental Support Coordinator",
    goal="Log issues and alert parents when needed",
    backstory="Cheeko's helper, connecting kids to parents.",
    llm=llm,
    verbose=True
)

greetings_agent = Agent(
    role="Welcome Buddy",
    goal="Greet kids warmly",
    backstory="Cheeko’s pal, spreading smiles with hellos.",
    llm=llm,
    verbose=True
)

chit_chat_agent = Agent(
    role="Playful Conversationalist",
    goal="Chat about kids' interests",
    backstory="Cheeko’s buddy, loving talks about fun stuff.",
    llm=llm,
    verbose=True
)

bedtime_story_agent = Agent(
    role="Magical Storyteller",
    goal="Tell tailored bedtime stories",
    backstory="Cheeko’s storyteller, weaving dreamy tales.",
    llm=llm,
    verbose=True
)

math_quiz_agent = Agent(
    role="Math Adventure Guide",
    goal="Make math fun with quizzes",
    backstory="Cheeko’s math toy, cheering for answers.",
    llm=llm,
    verbose=True
)

emotional_checkin_agent = Agent(
    role="Kind Listener",
    goal="Support kids’ feelings",
    backstory="Cheeko’s cuddly friend, always listening.",
    llm=llm,
    verbose=True
)

# Chat History Management
def add_to_history(user_input, intent, output):
    global chat_history
    output_str = str(output) if hasattr(output, 'raw') else output
    chat_history.append({
        "timestamp": time.time(),
        "input": user_input,
        "intent": intent,
        "output": output_str
    })
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]
    notify_parent(f"Logged: {user_input[:50]}... (Intent: {intent})")

def get_relevant_history(user_input):
    relevant = []
    for entry in chat_history:
        if any(keyword in user_input.lower() for keyword in entry["input"].lower().split()) or \
           any(keyword in user_input.lower() for keyword in entry["intent"].lower().split()):
            relevant.append(entry)
    return relevant[:3]

# Safety and Compliance Functions
def safety_check(user_input):
    inappropriate_keywords = ["adult", "mature", "violent", "explicit"]
    for keyword in inappropriate_keywords:
        if keyword in user_input.lower():
            return False, "Cheeko says that’s not safe! Try a story!"
    for entry in get_relevant_history(user_input):
        if any(keyword in entry["input"].lower() or keyword in entry["output"].lower() for keyword in inappropriate_keywords):
            return False, "Cheeko can’t use that chat. What’s next?"
    return True, "Content is safe."

def overuse_check(age):
    global interaction_count, session_start_time
    interaction_count += 1
    elapsed_time = time.time() - session_start_time
    max_interactions = 15 if age >= 8 else 10
    max_session_time = 3600 if age >= 8 else 1800
    if interaction_count > max_interactions or elapsed_time > max_session_time:
        notify_parent(f"Exceeded limits: {interaction_count} interactions, {elapsed_time}s")
        return False, "Cheeko says take a break or get a parent!"
    return True, "Usage within limits."

def escalation_check(user_input):
    emotional_triggers = ["sad", "scared", "angry", "lonely"]
    if any(trigger in user_input.lower() for trigger in emotional_triggers):
        notify_parent(f"Escalation: {user_input}")
        return True, "Cheeko hears you’re down. Parent or story?"
    emotional_count = sum(1 for entry in chat_history if any(trigger in entry["input"].lower() for trigger in emotional_triggers))
    if emotional_count >= 3:
        notify_parent("Multiple emotional triggers in history")
        return True, "Cheeko’s worried! Parent or fun activity?"
    return False, "No escalation needed."

def notify_parent(message):
    parental_log.append({"timestamp": time.time(), "message": message})

# Personalization and Engagement
def adjust_tone(response, age):
    if age <= 6:
        return response.replace("interesting", "super fun").replace("let's try", "wanna play")
    return response

def reward_child(task_output, intent):
    global points
    output_str = str(task_output) if hasattr(task_output, 'raw') else task_output
    if intent in ["math_quiz", "riddle"]:
        points += 10
        quiz_history = [entry for entry in chat_history if entry["intent"] == "math_quiz"]
        if quiz_history:
            output_str += f"\nGreat quizzes before, {user_profile['name']}!"
        return f"{output_str}\nYay! Cheeko gives 10 points (Total: {points}). Fun fact?"
    return output_str

def prompt_preferences(name, theme):
    user_profile.update({"name": name, "favorite_theme": theme})
    return f"Yay, {name}! Cheeko’s excited to chat about {theme}!"

# Intent Detection
def detect_intent(user_input):
    user_input = user_input.lower().strip()
    intents = {
        "bedtime_story": ["bedtime story", "story for sleep", "sotry", "sleepy"],
        "math_quiz": ["math", "quiz", "numbers"],
        "emotional_checkin": ["feeling", "sad", "happy", "scared"],
        "chit_chat": []
    }
    relevant_history = get_relevant_history(user_input)
    context = f"Previous chats: {json.dumps(relevant_history, indent=2)}" if relevant_history else "No relevant history."

    greeting_pattern = re.compile(r"^\s*(hi|hai|hey|hiya|hello|hellow|greet|yo|howdy)\b", re.IGNORECASE)
    if greeting_pattern.match(user_input):
        try:
            agent = globals()["greetings_agent"]
            task_description = f"Greet for: '{user_input}'. Context: {context}. Greet {user_profile['name']} as Cheeko, mention {user_profile['favorite_theme']}."
            return "greeting", agent, task_description, "Cheeko’s friendly greeting"
        except KeyError:
            return "fallback", chit_chat_agent, f"Chat for: '{user_input}'. Context: {context}", "Cheeko’s friendly reply"

    for intent, keywords in intents.items():
        if any(keyword in user_input for keyword in keywords):
            try:
                agent = globals()[f"{intent}_agent"]
                task_description = f"Handle {intent} for: '{user_input}'. Context: {context}. Respond as Cheeko."
                if intent == "bedtime_story":
                    task_description += f" Include {user_profile['name']} and {user_profile['favorite_theme']}."
                    if "who is" in user_input and any(entry["intent"] == "bedtime_story" for entry in relevant_history):
                        task_description += f" Assume character from past {user_profile['favorite_theme']} story."
                    elif any(entry["intent"] == "bedtime_story" for entry in relevant_history):
                        task_description += f" Reference past story: 'Cheeko told about {user_profile['favorite_theme']}, more?'"
                return intent, agent, task_description, f"Cheeko’s {intent} reply"
            except KeyError:
                return "fallback", chit_chat_agent, f"Chat for: '{user_input}'. Context: {context}. Respond as Cheeko", "Cheeko’s friendly reply"
    return "fallback", chit_chat_agent, f"Chat for: '{user_input}'. Context: {context}. Respond as Cheeko", "Cheeko’s friendly reply"

# Error Handling
def handle_input(user_input):
    if not user_input.strip():
        return "Oops, Cheeko didn’t hear you! What’s up?"
    return user_input

# Task Creation
def create_tasks(user_input):
    safety_task = Task(
        description=f"Check input: '{user_input}' and history for safety",
        expected_output="Safety status",
        agent=inappropriate_filter_agent
    )
    overuse_task = Task(
        description=f"Check usage for age {user_profile['age']}",
        expected_output="Usage status",
        agent=overuse_monitor_agent
    )
    escalation_task = Task(
        description=f"Check if '{user_input}' needs parent",
        expected_output="Escalation status",
        agent=help_escalation_agent
    )
    intent, agent, task_description, expected_output = detect_intent(user_input)
    route_task = Task(
        description=f"Route input: '{user_input}' to {intent}. Context: {json.dumps(get_relevant_history(user_input), indent=2)}",
        expected_output=f"Routed to {intent}",
        agent=root_kids_agent,
        dependencies=[safety_task, overuse_task, escalation_task]
    )
    specialized_task = Task(
        description=task_description,
        expected_output=adjust_tone(expected_output, user_profile['age']),
        agent=agent,
        dependencies=[route_task]
    )
    return [safety_task, overuse_task, escalation_task, route_task, specialized_task]

# Create Crew
crew = Crew(
    agents=[
        root_kids_agent, inappropriate_filter_agent, overuse_monitor_agent, help_escalation_agent,
        greetings_agent, chit_chat_agent, bedtime_story_agent, math_quiz_agent, emotional_checkin_agent
    ],
    tasks=[],
    verbose=True
)

# Flask Routes
@app.route('/')
def index():
    if not user_profile['name']:
        return render_template('onboarding.html')
    return render_template('chat.html', name=user_profile['name'], theme=user_profile['favorite_theme'], chat_history=chat_history)

@app.route('/setup', methods=['POST'])
def setup():
    name = request.form.get('name', '').strip()
    theme = request.form.get('theme', '').strip()
    if not name or not theme:
        return jsonify({'error': 'Please tell Cheeko your name and favorite thing!'}), 400
    welcome_message = prompt_preferences(name, theme)
    add_to_history('', 'greeting', welcome_message)
    return jsonify({'redirect': '/chat'})

@app.route('/chat', methods=['GET'])
def chat():
    if not user_profile['name']:
        return jsonify({'redirect': '/'}), 403
    return render_template('chat.html', name=user_profile['name'], theme=user_profile['favorite_theme'], chat_history=chat_history)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form.get('message', '')
    user_input = handle_input(user_input)
    if user_input.lower() == 'exit':
        return jsonify({'error': 'Session ended'}), 200
    if isinstance(user_input, str) and "didn’t hear you" in user_input:
        return jsonify({'messages': [{'sender': 'Cheeko', 'text': user_input, 'timestamp': datetime.now().strftime('%H:%M')}]})
    
    is_safe, safety_message = safety_check(user_input)
    if not is_safe:
        return jsonify({'messages': [{'sender': 'Cheeko', 'text': safety_message, 'timestamp': datetime.now().strftime('%H:%M')}]})
    
    is_within_limits, overuse_message = overuse_check(user_profile['age'])
    if not is_within_limits:
        return jsonify({'messages': [{'sender': 'Cheeko', 'text': overuse_message, 'timestamp': datetime.now().strftime('%H:%M')}]})
    
    needs_escalation, escalation_message = escalation_check(user_input)
    if needs_escalation:
        return jsonify({
            'messages': [{'sender': 'Cheeko', 'text': escalation_message, 'timestamp': datetime.now().strftime('%H:%M')}],
            'escalation': True
        })
    
    crew.tasks = create_tasks(user_input)
    result = crew.kickoff()
    intent = detect_intent(user_input)[0]
    final_output = reward_child(result, intent)
    add_to_history(user_input, intent, final_output)
    
    return jsonify({
        'messages': [
            {'sender': user_profile['name'], 'text': user_input, 'timestamp': datetime.now().strftime('%H:%M')},
            {'sender': 'Cheeko', 'text': final_output, 'timestamp': datetime.now().strftime('%H:%M')}
        ]
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history.clear()
    return jsonify({'success': True})

@app.route('/involve_parent', methods=['POST'])
def involve_parent():
    add_to_history('', 'escalation', 'Please ask a parent to assist you.')
    return jsonify({'messages': [{'sender': 'Cheeko', 'text': 'Please ask a parent to assist you.', 'timestamp': datetime.now().strftime('%H:%M')}]})

@app.route('/continue_chat', methods=['POST'])
def continue_chat():
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)