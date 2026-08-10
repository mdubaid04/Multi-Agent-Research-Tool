from agents import build_readerAgent,build_searchAgent,writter_chain,critics_chain
from datetime import date
today=date.today().strftime("%B-%d-%Y")

def run_pipeline(topic:str)->dict:
  state={}
  print("\n"+" ="*50)
  print("step 1 - search agent is working ...")
  print("="*50)
  # step -> 1  search agent -----------------------------
  searchAgent=build_searchAgent()
  response=searchAgent.invoke({'messages':[("user", f"Today's date is {today}.\n Find recent, reliable and detailed information about: {topic}.\n Always use the search tool results for current information, never rely on your own knowledge for recent events..")]})
  state["search_result"]=response["messages"][-1].content
  print("\n search result ",state['search_result'])

  #  step -> 2  reader agent ----------------------------

  print("\n"+" ="*50)
  print("step 2 - reader agent is working ...")
  print("="*50)

  readerAgent=build_readerAgent()
  response=readerAgent.invoke({"messages": [("user",
            f"Based on the following search result about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Result:\n{state['search_result'][:800]}"
            "You must call the scrapeTextFromUrl tool exactly once with the most relevent URL Do not attempt multiple scrapes. If the result is insufficient, work with what you have and return the best you can."
        )]})

  state["scraped_content"]=response["messages"][-1].content
  print("\n scraped content ",state['scraped_content'])

  #  step -> 3  writer Chain ----------------------------

  print("\n"+" ="*50)
  print("step 3 - writer agent is working ...")
  print("="*50)

  research_combined={
     f"SEARCH RESULT : \n {state['search_result']} \n\n"
     f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
  }

  state["report"]=writter_chain.invoke({"topic":topic,"research":research_combined})

  print("\n writer report \n", state['report'])

  #  step -> 4  Critics Chain ----------------------------

  print("\n"+" ="*50)
  print("step 3 - writer agent is working ...")
  print("="*50)

  state["feedback"]=critics_chain.invoke({"report":state["report"]})

  print("\n critic report \n", state['feedback'])
  return state


if __name__ == "__main__":  
  topic=input("Enter topic: ")
  run_pipeline(topic)
  