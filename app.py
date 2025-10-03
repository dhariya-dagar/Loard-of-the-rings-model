from langchain.vectorstores import Chroma
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain_community.tools import BraveSearch
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
load_dotenv()
import os
# loading the api keys from the .env
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['BRAIN_API_KEY']=os.getenv("BRAIN_API_KEY")

st.title("Loard of the Rings Chatbot")



# defining embidding model
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004")
# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)
# setting as reteriver
retriver=db.as_retriever()


# creating wrapper over wiki api
api_wrapper = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

websearch = BraveSearch.from_api_key(search_kwargs={"count": 3})

# tools
tools=[retriver,wiki,websearch]

# setting up llm
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)

# tools prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# creating agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

queary = st.text_input("input your question here")

result = agent_executor.invoke({"input":queary})
st.write(result)