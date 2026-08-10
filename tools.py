from langchain_core.tools import tool
from bs4 import BeautifulSoup
from tavily import TavilyClient
import requests
from dotenv import load_dotenv
import os

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
@tool
def searchQueryTool(query:str)->str:
  """Search the web for recent and reliable information on a topic.Returns Titles,URLs and Snippets."""
  tavily=TavilyClient(api_key=tavily_api_key)
  search_result=tavily.search(query=query,max_results=5,days=1,language="en")
  out=[]
  for r in search_result["results"]:
    out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")
  return "\n----\n".join(out)

@tool
def scrapeTextFromUrl(url)->str:
   """Scrape and return clean text content from a given URL for deeper reading."""
   try:
    response=requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
    soup=BeautifulSoup(response.text,"html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
      tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:3000]  
   except Exception as e:
    return  f"Could not scrape URL: {str(e)}"