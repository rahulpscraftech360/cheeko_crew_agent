import tkinter as tk
from tkinter import ttk
import json
import time
from crewai import Agent, Task, Crew, LLM
import re

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

# Tkinter UI with WhatsApp-style Enhancements
class CheekoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cheeko Chat Setup")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Onboarding frame
        self.onboarding_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.onboarding_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.onboarding_frame, text="Hi! I’m Cheeko, your fun toy buddy!", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)
        
        tk.Label(self.onboarding_frame, text="What’s your name?", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
        self.name_entry = tk.Entry(self.onboarding_frame, font=("Arial", 12))
        self.name_entry.pack(pady=5)

        tk.Label(self.onboarding_frame, text="What’s your favorite thing (like dragons or superheroes)?", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
        self.theme_entry = tk.Entry(self.onboarding_frame, font=("Arial", 12))
        self.theme_entry.pack(pady=5)

        tk.Button(self.onboarding_frame, text="Start Chatting with Cheeko", font=("Arial", 12), bg="#128C7E", fg="white", command=self.setup_profile).pack(pady=20)

    def setup_profile(self):
        name = self.name_entry.get().strip()
        theme = self.theme_entry.get().strip()
        if not name or not theme:
            tk.Label(self.onboarding_frame, text="Please tell Cheeko your name and favorite thing!", fg="red", bg="#f0f0f0").pack(pady=5)
            return
        user_profile.update({"name": name, "favorite_theme": theme})
        self.onboarding_frame.destroy()
        self.create_chat_ui()
        self.display_message(f"Cheeko: {prompt_preferences(name, theme)}", is_user=False)

    def create_chat_ui(self):
        self.root.title(f"Cheeko’s {user_profile['favorite_theme'].capitalize()} Chat with {user_profile['name']}")

        # Header
        self.header = tk.Frame(self.root, bg="#128C7E", height=60)
        self.header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(self.header, text="Cheeko Chat", font=("Arial", 16, "bold"), bg="#128C7E", fg="white").pack(side=tk.LEFT, pady=10, padx=10)
        tk.Button(self.header, text="Clear Chat", font=("Arial", 12), bg="#128C7E", fg="white", command=self.clear_chat).pack(side=tk.RIGHT, padx=10)

        # Main frame
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Chat list (left panel)
        self.chat_list = tk.Listbox(self.main_frame, width=25, bg="#ffffff", font=("Arial", 12))
        self.chat_list.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.chat_list.insert(tk.END, "Cheeko’s Chat")
        self.chat_list.select_set(0)

        # Chat window (scrollable canvas)
        self.chat_frame = tk.Frame(self.main_frame, bg="#ECE5DD")
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat_canvas = tk.Canvas(self.chat_frame, bg="#ECE5DD", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.message_frame = tk.Frame(self.chat_canvas, bg="#ECE5DD")
        self.chat_canvas.create_window((0, 0), window=self.message_frame, anchor="nw")
        self.message_frame.bind("<Configure>", lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Input area
        self.input_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.input_field = tk.Entry(self.input_frame, font=("Arial", 12))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send to Cheeko", font=("Arial", 12), bg="#128C7E", fg="white", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)

    def _on_mousewheel(self, event):
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def display_message(self, message, is_user=False):
        bubble_frame = tk.Frame(self.message_frame, bg="#ECE5DD")
        bubble_frame.pack(fill=tk.X, padx=10, pady=5)

        if is_user:
            bubble_bg = "#DCF8C6"  # Green for user
            bubble_frame.pack(anchor="e")
            bubble_width = 50
            anchor = "e"
        else:
            bubble_bg = "#FFFFFF"  # White for Cheeko
            bubble_frame.pack(anchor="w")
            bubble_width = 50
            anchor = "w"

        message_label = tk.Label(
            bubble_frame,
            text=message,
            font=("Arial", 12),
            bg=bubble_bg,
            wraplength=400,
            justify="left" if not is_user else "right",
            padx=10,
            pady=5
        )
        message_label.pack(anchor=anchor)
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1)

    def clear_chat(self):
        for widget in self.message_frame.winfo_children():
            widget.destroy()
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1)

    def send_message(self, event=None):
        user_input = handle_input(self.input_field.get())
        self.input_field.delete(0, tk.END)
        self.display_message(f"{user_profile['name']}: {user_input}", is_user=True)
        if user_input.lower() == 'exit':
            self.root.quit()
            return
        if isinstance(user_input, str) and "didn’t hear you" in user_input:
            self.display_message(f"Cheeko: {user_input}", is_user=False)
            return
        is_safe, safety_message = safety_check(user_input)
        if not is_safe:
            self.display_message(f"Cheeko: {safety_message}", is_user=False)
            return
        is_within_limits, overuse_message = overuse_check(user_profile['age'])
        if not is_within_limits:
            self.display_message(f"Cheeko: {overuse_message}", is_user=False)
            return
        needs_escalation, escalation_message = escalation_check(user_input)
        if needs_escalation:
            self.display_message(f"Cheeko: {escalation_message}", is_user=False)
            self.input_field.config(state='disabled')
            tk.Button(self.input_frame, text="Yes, parent", command=self.involve_parent).pack(side=tk.RIGHT, padx=5)
            tk.Button(self.input_frame, text="No, continue", command=self.continue_chat).pack(side=tk.RIGHT, padx=5)
            return
        loading_label = tk.Label(self.message_frame, text="Cheeko is typing...", font=("Arial", 10, "italic"), bg="#ECE5DD", fg="#555555")
        loading_label.pack(anchor="w", padx=10, pady=5)
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1)
        self.root.update()

        crew.tasks = create_tasks(user_input)
        result = crew.kickoff()
        intent = detect_intent(user_input)[0]
        final_output = reward_child(result, intent)

        loading_label.destroy()
        self.display_message(f"Cheeko: {final_output}", is_user=False)
        add_to_history(user_input, intent, final_output)

    def involve_parent(self):
        self.display_message("Cheeko: Ask a parent to help!", is_user=False)
        self.input_field.config(state='normal')
        for widget in self.input_frame.winfo_children():
            if isinstance(widget, tk.Button) and widget != self.send_button:
                widget.destroy()

    def continue_chat(self):
        self.input_field.config(state='normal')
        for widget in self.input_frame.winfo_children():
            if isinstance(widget, tk.Button) and widget != self.send_button:
                widget.destroy()

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = CheekoUI(root)
    root.mainloop()