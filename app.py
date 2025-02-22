import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain_community.graphs import Neo4jGraph

from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine #To map the output which is coming from database.
import sqlite3
from langchain.chains import GraphCypherQAChain
from sqlalchemy.exc import OperationalError
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="LANGCHAIN:CHAT WITH DATABASE", page_icon=":robot_face:")
st.title("Langchain:Chat with DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"
NEO4J="USE_NEO4J"

#Example prompt for GraphDB
examples=[
    {
        "question":"Find the Actor with the Highest IMDb Rating Across All Movies They’ve Acted In",
        "query":"MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) WITH a, AVG(m.imdbRating) AS avgRating RETURN a.name AS Actor, avgRating ORDER BY avgRating DESC LIMIT 1"
    },
    {
        "question": "Find the Most Common Genre Among All Movies",
        "query": "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) RETURN g.name AS Genre, COUNT(m) AS MovieCount ORDER BY MovieCount DESC LIMIT 1"
    },
    {
        "question": "Find the Actor Who Has Acted in the Most Number of Movies",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name AS Actor, COUNT(m) AS MovieCount ORDER BY MovieCount DESC LIMIT 1"
    },
    {
        "question": "Find Pairs of Actors Who Have Acted Together in the Most Movies",
        "query": "MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor) WHERE a1 <> a2 RETURN a1.name AS Actor1, a2.name AS Actor2, COUNT(m) AS MovieCount ORDER BY MovieCount DESC LIMIT 5"
    },
    {
        "question": "Find the Year with the Highest Number of Movies Released",
        "query": "MATCH (m:Movie) RETURN m.released.year AS Year, COUNT(m) AS MovieCount ORDER BY MovieCount DESC LIMIT 1"
    },
    {
        "question": "Find Actors Who Have Acted in Both Action and Comedy Movies",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Action' WITH a MATCH (a)-[:ACTED_IN]->(m2:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g2.name = 'Comedy' RETURN DISTINCT a.name AS Actor"
    },
    {
        "question": "Find the Actor Who Has Appeared in Movies Across the Most Number of Genres, Along with Their Average IMDb Rating",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre) WITH a, COLLECT(DISTINCT g.name) AS genres, AVG(m.imdbRating) AS avgRating RETURN a.name AS Actor, SIZE(genres) AS genreCount, avgRating ORDER BY genreCount DESC, avgRating DESC LIMIT 1"
    },
    {
        "question": "Find the Actor Who Has Acted in the Most Movies with an IMDb Rating Above 8 and Also Appeared in the Most Genres",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre) WHERE m.imdbRating > 8 WITH a, COUNT(m) AS movieCount, COLLECT(DISTINCT g.name) AS genres RETURN a.name AS Actor, movieCount, SIZE(genres) AS genreCount ORDER BY movieCount DESC, genreCount DESC LIMIT 1"
    },
    {
        "question": "List All Directors Who Have Directed More Than 3 Movies with an IMDb Rating Above 7 and Belong to Multiple Genres",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie)-[:IN_GENRE]->(g:Genre) WHERE m.imdbRating > 7 WITH p, COUNT(m) AS movieCount, COLLECT(DISTINCT g.name) AS genres WHERE movieCount > 3 AND SIZE(genres) > 1 RETURN p.name AS Director, movieCount, SIZE(genres) AS genreCount"
    }
    
  ]


radio_opt=["Use SQLLite 3 Database","Connect to your SQL Database","Connect to Neo4j Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Hostname")
    mysql_user=st.sidebar.text_input("Provide MySQL Username")
    mysql_password=st.sidebar.text_input("Provide MySQL Password",type="password")
    mysql_db=st.sidebar.text_input("Provide MySQL Database Name")
elif radio_opt.index(selected_opt)==2:
    db_uri=NEO4J
    neo4j_username=st.sidebar.text_input("Provide Neo4j Username")
    neo4j_password=st.sidebar.text_input("Provide Neo4j Password",type="password")
    neo4j_url=st.sidebar.text_input("Provide Neo4j URL")
    
else:
    db_uri=LOCALDB
    
api_key=st.sidebar.text_input(label="GROQ API Key",type="password")

if not db_uri:
    st.info("Please neter the database information and uri") 
    
if not api_key:
    st.info("Please enter your GROQ API key")

## LLM MODEL
llm=ChatGroq(groq_api_key=api_key,model_name="qwen-2.5-coder-32b",streaming=True)


@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None): 
    """Connecting to database"""
    try:
        if db_uri == LOCALDB:
            dbfilepath = (Path(__file__).parent / "student.db").absolute()
            print(dbfilepath)  # Print the database file path for debugging
            
            # Check if the SQLite file exists before trying to connect
            if not dbfilepath.exists():
                raise FileNotFoundError("SQLite database file not found.")

            # Creates a database engine that allows Python to interact with the database
            creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))

        elif db_uri == MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details")
                st.stop()
            
            print(f"mysql_host: {mysql_host}")  # Print MySQL hostname for debugging
            
            # Creates a database engine that allows Python to interact with the database
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

        elif db_uri == NEO4J:
            if not (neo4j_username and neo4j_password and neo4j_url):
                st.error("Please provide all Neo4j connection details")
                st.stop()
            
            print(f"neo4j_url: {neo4j_url}")  # Print Neo4j URL for debugging
            
            return Neo4jGraph(
                url=neo4j_url, username=neo4j_username, password=neo4j_password
            )  

    except FileNotFoundError as e:
        # Display user-friendly error message if SQLite file is missing
        st.error("Error: The SQLite database file is missing. Please check the file path.")
        st.stop()

    except OperationalError as e:
        # Display user-friendly error message if database cannot be opened
        st.error("Error: Unable to open the database. Please check if the database exists and is accessible.")
        st.stop()

    except Exception as e:
        # Catch any other unexpected errors and display them safely
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()
    
    return db  # Ensures function always returns a valid database connection or stops execution


if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
elif db_uri==NEO4J:
    db = configure_db(db_uri,None,neo4j_username,neo4j_password, neo4j_url)

else:
    db=configure_db(db_uri)
    

#Toolkit
if db_uri == NEO4J:
    example_prompt=PromptTemplate.from_template(
    "User input:{question}\n Cypher query:{query}"
    )

    prompt=FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="You are a Neo4j expert. Given an input question,create a syntactically very accurate Cypher query",
        suffix="User input: {question}\nCypher query: ",
        input_variables=["question","schema"]
    )
    agent = GraphCypherQAChain.from_llm(graph=db,llm=llm,verbose=True,allow_dangerous_requests=True,cypher_prompt=prompt)
else:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
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
    
   
    
    try:
        with st.chat_message("assistant"):  # When the agent is communicating
            streamlit_callback = StreamlitCallbackHandler(st.container())  # Chain of thoughts
            response = agent.run(user_query, callbacks=[streamlit_callback])  # Run the agent
            st.session_state.messages.append({"role": "assistant", "content": response})  # Store message first
            st.write(response)  # Then display it on frontend
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": "Please provide a relevant question to fetch data."})
        st.chat_message("assistant").write("Please provide a relevant question to fetch data.")
