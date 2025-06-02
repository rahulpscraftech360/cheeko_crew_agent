from crewai import Agent, Task, Crew, LLM
import re
import time
import json

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
user_profile = {"name": "Emma", "age": 7, "favorite_theme": "dragons"}
points = 0
parental_log = []
chat_history = []  # Store input-output pairs
MAX_HISTORY = 20  # Limit history size

# Define all agents (same as original)
root_kids_agent = Agent(
    role="Central Controller",
    goal="Analyze input, use chat history, ensure safety, and route tasks",
    backstory="A friendly, safety-first coordinator that remembers past chats to make conversations fun and continuous for kids",
    llm=llm,
    verbose=True
)
inappropriate_filter_agent = Agent(
    role="Content Safety Monitor",
    goal="Block unsafe content in inputs and history",
    backstory="A vigilant guardian ensuring all interactions and past references are safe",
    llm=llm,
    verbose=True
)
overuse_monitor_agent = Agent(
    role="Usage Limiter",
    goal="Limit screen time",
    backstory="Encourages balanced usage",
    llm=llm,
    verbose=True
)
help_escalation_agent = Agent(
    role="Escalation Coordinator",
    goal="Suggest parental supervision and log history",
    backstory="Knows when parents should step in and keeps interaction records",
    llm=llm,
    verbose=True
)
greetings_agent = Agent(
    role="Greetings Specialist",
    goal="Welcome kids",
    backstory="A cheerful greeter",
    llm=llm,
    verbose=True
)
chit_chat_agent = Agent(
    role="Small Talk Facilitator",
    goal="Engage in fun chats",
    backstory="A playful conversationalist",
    llm=llm,
    verbose=True
)
bedtime_story_agent = Agent(
    role="Bedtime Storyteller",
    goal="Tell soothing stories",
    backstory="A gentle narrator",
    llm=llm,
    verbose=True
)
math_quiz_agent = Agent(
    role="Math Quiz Master",
    goal="Conduct math quizzes",
    backstory="A fun math enthusiast",
    llm=llm,
    verbose=True
)
emotional_checkin_agent = Agent(
    role="Emotional Support Guide",
    goal="Support emotional expression",
    backstory="A compassionate listener",
    llm=llm,
    verbose=True
)

# Chat History Management
def add_to_history(user_input, intent, output):
    global chat_history
    # Convert output to string to avoid CrewOutput issues
    output_str = str(output) if hasattr(output, 'raw') else output
    chat_history.append({
        "timestamp": time.time(),
        "input": user_input,
        "intent": intent,
        "output": output_str
    })
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]  # Keep last 20 interactions
    notify_parent(f"Logged interaction: {user_input[:50]}... (Intent: {intent})")

def get_relevant_history(user_input):
    # Find relevant history based on keywords or intent
    relevant = []
    for entry in chat_history:
        if any(keyword in user_input.lower() for keyword in entry["input"].lower().split()) or \
           any(keyword in user_input.lower() for keyword in entry["intent"].lower().split()):
            relevant.append(entry)
    return relevant[:3]  # Return up to 3 relevant entries

# Safety and Compliance Functions
def safety_check(user_input):
    inappropriate_keywords = ["adult", "mature", "violent", "explicit"]
    for keyword in inappropriate_keywords:
        if keyword in user_input.lower():
            return False, "Sorry, that request isn't safe for kids. Try something fun like a story!"
    # Check history for inappropriate content
    for entry in get_relevant_history(user_input):
        if any(keyword in entry["input"].lower() or keyword in entry["output"].lower() for keyword in inappropriate_keywords):
            return False, "Sorry, I can't use that past chat because it's not safe. What's next?"
    return True, "Content is safe."

def overuse_check(age):
    global interaction_count, session_start_time
    interaction_count += 1
    elapsed_time = time.time() - session_start_time
    max_interactions = 15 if age >= 8 else 10
    max_session_time = 3600 if age >= 8 else 1800
    if interaction_count > max_interactions or elapsed_time > max_session_time:
        notify_parent(f"Child exceeded usage limits: {interaction_count} interactions, {elapsed_time} seconds")
        return False, "You've been chatting a lot! Let's take a break or ask a parent to join."
    return True, "Usage within limits."

def escalation_check(user_input):
    emotional_triggers = ["sad", "scared", "angry", "lonely"]
    if any(trigger in user_input.lower() for trigger in emotional_triggers):
        notify_parent(f"Escalation triggered for input: {user_input}")
        return True, "It sounds like you're feeling down. Want to talk to a parent or hear a fun story?"
    # Check history for repeated emotional triggers
    emotional_count = sum(1 for entry in chat_history if any(trigger in entry["input"].lower() for trigger in emotional_triggers))
    if emotional_count >= 3:
        notify_parent("Multiple emotional triggers detected in history")
        return True, "You've mentioned feeling down a few times. Want to talk to a parent or try a fun activity?"
    return False, "No escalation needed."

def notify_parent(message):
    parental_log.append({"timestamp": time.time(), "message": message})
    print(f"Parental Notification: {message}")  # Placeholder for real notification

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
        # Check history for past quiz performance
        quiz_history = [entry for entry in chat_history if entry["intent"] == "math_quiz"]
        if quiz_history:
            output_str += f"\nYou did great on quizzes before, {user_profile['name']}!"
        return f"{output_str}\nGreat job! You earned 10 points (Total: {points}). Want a fun fact as a reward?"
    return output_str

def prompt_preferences():
    name = input("Hi! What's your name? ")
    theme = input("What's your favorite thing, like animals or superheroes? ")
    user_profile.update({"name": name, "favorite_theme": theme})
    return f"Awesome, {name}! I'll use {theme} in our chats!"

# Intent Detection with History
def detect_intent(user_input):
    user_input = user_input.lower()
    intents = {
        "bedtime_story": ["bedtime story", "story for sleep", "sotry", "sleepy"],  # Added "sotry", "sleepy"
        "greeting": ["hello", "greet", "hi"],
        "math_quiz": ["math", "quiz", "numbers"],
        "emotional_checkin": ["feeling", "sad", "happy", "scared"],
        "chit_chat": []
    }
    relevant_history = get_relevant_history(user_input)
    context = f"Previous chats: {json.dumps(relevant_history, indent=2)}" if relevant_history else "No relevant history."
    for intent, keywords in intents.items():
        if any(keyword in user_input for keyword in keywords):
            agent = globals()[f"{intent}_agent"]
            task_description = f"Handle {intent} for input: '{user_input}'. Context: {context}"
            if intent == "bedtime_story":
                task_description += f" Include {user_profile['name']} and {user_profile['favorite_theme']}."
                # Check for character references like "fluffy"
                if "who is" in user_input and any(entry["intent"] == "bedtime_story" for entry in relevant_history):
                    task_description += f" If asking about a character (e.g., 'Fluffy'), assume it's from a previous {user_profile['favorite_theme']} story and provide details."
                elif any(entry["intent"] == "bedtime_story" for entry in relevant_history):
                    task_description += f" Reference past bedtime story, e.g., 'Last time you loved {user_profile['favorite_theme']}, want more?'"
            return intent, agent, task_description, f"A {intent} response"
    return "fallback", chit_chat_agent, f"Engage in friendly chat for: '{user_input}'. Context: {context}", "A friendly response"

# Error Handling
def handle_input(user_input):
    if not user_input.strip():
        return "Oops, I didn't hear you! What do you want to talk about?"
    return user_input

# Task Creation
def create_tasks(user_input):
    safety_task = Task(
        description=f"Review input: '{user_input}' and history for inappropriate content",
        expected_output="Content safety status",
        agent=inappropriate_filter_agent
    )
    overuse_task = Task(
        description=f"Check usage limits for age {user_profile['age']}",
        expected_output="Usage status",
        agent=overuse_monitor_agent
    )
    bedtime_task = Task(
        description=f"Check usage limits for age {user_profile['age']}",
        expected_output="Usage status",
        agent=overuse_monitor_agent
    )
    escalation_task = Task(
        description=f"Evaluate if '{user_input}' or history needs parental supervision",
        expected_output="Escalation status",
        agent=help_escalation_agent
    )
    intent, agent, task_description, expected_output = detect_intent(user_input)
    route_task = Task(
        description=f"Route input: '{user_input}' to the {intent} agent",
        expected_output=f"Task routed to {intent}",
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

# Interactive Loop
print(prompt_preferences())  # Prompt for name and theme at start
while True:
    user_input = handle_input(input("Enter your request (or type 'exit' to quit): "))
    if user_input.lower() == 'exit':
        break
    if isinstance(user_input, str) and "didn't hear you" in user_input:
        print(user_input)
        continue
    is_safe, safety_message = safety_check(user_input)
    if not is_safe:
        print(safety_message)
        continue
    is_within_limits, overuse_message = overuse_check(user_profile['age'])
    if not is_within_limits:
        print(overuse_message)
        continue
    needs_escalation, escalation_message = escalation_check(user_input)
    if needs_escalation:
        print(escalation_message)
        user_response = input("Type 'yes' to involve a parent, or 'no' to continue: ")
        if user_response.lower() == 'yes':
            print("Please ask a parent to assist you.")
            continue
    crew.tasks = create_tasks(user_input)
    result = crew.kickoff()
    intent = detect_intent(user_input)[0]
    final_output = reward_child(result, intent)
    print(final_output)
    add_to_history(user_input, intent, final_output)
