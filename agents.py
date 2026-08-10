from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tools import searchQueryTool,scrapeTextFromUrl
import os
load_dotenv()

mystral_agent=ChatMistralAI(model="mistral-small-latest",api_key=os.getenv("MISTRAL_API_KEY"))
groq_llama_agent=ChatGroq(model="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY"))
groq_qwen_agent=ChatGroq(model="qwen/qwen3.6-27b",api_key=os.getenv("GROQ_API_KEY"),reasoning_format="hidden")




# response=mistra_agent.invoke("Hello, how are you?")
# print (response.content)
parser=StrOutputParser()

def build_searchAgent():
  return create_agent(
    model=mystral_agent,
    tools=[searchQueryTool],
  )

def build_readerAgent():
  return create_agent(
    model=groq_llama_agent,
    tools=[scrapeTextFromUrl],
  )

writerPrompt=ChatPromptTemplate.from_messages([
   ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writter_chain=writerPrompt | mystral_agent | parser

criticsPrompt=ChatPromptTemplate.from_messages([
   ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critics_chain=criticsPrompt | groq_qwen_agent | parser