import os
import asyncio

# ---- LLM API clients ----
from openai import OpenAI
from anthropic import Anthropic
import cohere
from cohere import Client as CohereClient

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))
co2 = CohereClient(api_key=os.getenv("COHERE_API_KEY"))


def chat(prompt):
    r = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def claude(prompt):
    m = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return m.content[0].text


# ---- Hugging Face models ----
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from huggingface_hub import InferenceClient

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
embed = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf = InferenceClient(token=os.getenv("HF_TOKEN"))

# not loaded yet, just keeping the reference here
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ---- Vertex / Gemini ----
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai

aiplatform.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")

gemini = GenerativeModel("gemini-1.5-pro")
vx = genai.Client(vertexai=True, project=os.getenv("GCP_PROJECT"), location="us-central1")

# our own tuned model served on a Vertex endpoint
endpoint = aiplatform.Endpoint(
    endpoint_name="projects/demo/locations/us-central1/endpoints/123"
)


# ---- core ML frameworks ----
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


keras_model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="relu")])
clf = LogisticRegression()


# ---- agent frameworks ----
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from autogen import AssistantAgent, UserProxyAgent
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

lc_llm = ChatOpenAI(model="gpt-4.1")
lc_agent = initialize_agent(tools=[], llm=lc_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

researcher = Agent(role="Researcher", goal="dig up facts", backstory="senior analyst")
crew = Crew(agents=[researcher], tasks=[Task(description="research the topic", agent=researcher)])

assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy")

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY")))


# ---- MCP server ----
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")


@mcp.tool()
def get_weather(city):
    return f"{city}: sunny"


@mcp.resource("config://app")
def get_config():
    return "app config"


# ---- MCP client ----
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])


async def call_tool():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool("get_weather", {"city": "Pune"})


if __name__ == "__main__":
    print(chat("hello"))
    asyncio.run(call_tool())
