import random
import uuid
from typing import List, Callable

class Agent:
    def __init__(self, prompt: str, generate_fn: Callable, evaluate_fn: Callable, mutate_fn: Callable):
        self.id = uuid.uuid4()
        self.prompt = prompt
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.mutate_fn = mutate_fn
        self.artifact = None
        self.score = 0

    def generate(self):
        self.artifact = self.generate_fn(self.prompt)

    def evaluate(self):
        self.score = self.evaluate_fn(self.artifact)

    def reflect(self, top_prompts: List[str]):
        # Blend top ideas (crossover) and mutate
        new_prompt = self.mutate_fn(self.prompt, top_prompts)
        self.prompt = new_prompt


class Population:
    def __init__(self, agents: List[Agent], top_k=3):
        self.agents = agents
        self.top_k = top_k

    def step(self):
        for agent in self.agents:
            agent.generate()
            agent.evaluate()

        # Sort by fitness score
        self.agents.sort(key=lambda x: x.score, reverse=True)
        top_prompts = [a.prompt for a in self.agents[:self.top_k]]

        for agent in self.agents:
            agent.reflect(top_prompts)

    def run(self, iterations=5):
        for i in range(iterations):
            print(f"\n=== Iteration {i+1} ===")
            self.step()
            for agent in self.agents:
                print(f"{agent.id} | Score: {agent.score:.2f} | Prompt: {agent.prompt}")


# === Abstracted LLM functions ===
def dummy_generate(prompt):
    return f"Generated output based on: {prompt}"

def dummy_evaluate(artifact):
    return random.uniform(0, 1)  # Replace with real scoring logic

def dummy_mutate(prompt, top_prompts):
    base = random.choice(top_prompts)
    mutation = random.choice(["extend", "refactor", "contrast"])
    return f"{base} + mutation: {mutation}"

# === Create a population ===
initial_prompts = [
    "How to teach AI ethics to teenagers?",
    "Design a sustainable city in the year 2100.",
    "Can an AI write poetry that moves people?",
    "Simulate a Mars colony power grid.",
]

agents = [Agent(prompt, dummy_generate, dummy_evaluate, dummy_mutate) for prompt in initial_prompts]
pop = Population(agents)
pop.run(iterations=5)
