import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine #To map the output which is coming from database.
import sqlite3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Langchain:Chat with SQL DB", page_icon=":robot_face:")
st.title("Langchain:Chat with SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"


radio_opt=["Use SQLLite 3 Database- Student.db","Connect to your SQL Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Hostname")
    mysql_user=st.sidebar.text_input("Provide MySQL Username")
    mysql_password=st.sidebar.text_input("Provide MySQL Password",type="password")
    mysql_db=st.sidebar.text_input("Provide MySQL Database Name")
else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="GROQ API Key",type="password")

if not db_uri:
    st.info("Please neter the database information and uri") 
    
if not api_key:
    st.info("Please enter your GROQ API key")

## LLM MODEL
llm=ChatGroq(groq_api_key=api_key,model_name="Qwen-2.5-32b",streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None): #Connecting to database
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator=lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))#creates a database engine that allows Python to interact with the database.
    elif db_uri==MYSQL:
        if not(mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySql connection details")
            st.stop()
        print(f"mysql_host: {mysql_host}")
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))  #creates a database engine that allows Python to interact with the database.
        
    return db

if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)
    

#Toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"): #If there are no messages or if user wants to clear the message history the chat history will be cleared
    st.session_state["messages"]=[{"role":"system","content":"How can I help you?"}]
    
for msg in st.session_state.messages:   #Displaying the chat history
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database") 

if user_query: #When user writes query and hits enter
    st.session_state.messages.append({"role":"user","content":user_query}) # Store message first
    st.chat_message("user").write(user_query)# Then display it on frontend
    
    
    with st.chat_message("assistant"): #When the agent is communicating with itself
        streamlit_callback=StreamlitCallbackHandler(st.container()) #Chain of thoughts
        response=agent.run(user_query,callbacks=[streamlit_callback]) #Run the agent
        st.session_state.messages.append({"role":"assistant","content":response}) # Store message first
        st.write(response) # Then display it on frontend