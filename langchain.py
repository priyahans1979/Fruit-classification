from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1. Initialize your LLM
groq_api = "YOUR_GROQ_API_KEY"

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api,
    temperature=0.1
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="suggest a 100 word blog on title about {topic}."
)

chain = prompt | llm

topic = input("enter a topic")
output = chain.invoke(topic)

print('Generated blog title", output.content')