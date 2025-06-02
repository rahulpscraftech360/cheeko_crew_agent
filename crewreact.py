from crewai import Agent, Task, Crew, LLM

# Initialize Large Language Model (LLM)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=0.9,
    stop=None,
    stream=False,
)

# Create CrewAI agents
summarizer = Agent(
    role='Documentation Summarizer',
    goal='Create concise summaries of technical documentation',
    backstory='Technical writer who excels at simplifying complex concepts',
    llm=llm,
    verbose=True
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to other languages',
    backstory='Technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

# Define tasks
summary_task = Task(
    description='Summarize this React hook documentation:\n\nuseFetch(url) is a custom hook for making HTTP requests. It returns { data, loading, error } and automatically handles loading states.',
    expected_output="A clear, concise summary of the hook's functionality",
    agent=summarizer
)

translation_task = Task(
    description='Translate the summary to Turkish',
    expected_output="Turkish translation of the hook documentation",
    agent=translator,
    dependencies=[summary_task]
)

# Create crew to manage agents and tasks
crew = Crew(
    agents=[summarizer, translator],
    tasks=[summary_task, translation_task],
    verbose=True
)

# Execute the crew
result = crew.kickoff()
print(result)