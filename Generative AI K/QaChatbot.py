'''Simpler langchain Streamlit app to demonstrate a QA chatbot
A beginner-friendly version focusing on the core concepts'''

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import os
from langchain.prompts import ChatPromptTemplate


##Page config
st.set_page_config(page_title="QA Chatbot with Cohere", page_icon="🤖")


##Title
st.title("🤖 QA Chatbot with Cohere & LangChain")
st.markdown("A simple QA chatbot using Cohere's chat model and LangChain.")

with st.sidebar:
    st.header("settings")

    ##api key
    api_key = st.text_input("Enter your Cohere API key:", type="password",help="You can get your API key from https://dashboard.cohere.com/api-keys")

    ##model selection
    model_name = st.selectbox("Select a Cohere model:", options=["c4ai-aya-vision-8b","c4ai-aya-vision-32b"], index=0, help="Select a Cohere chat model.")

    ##clear chat
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

##Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

##Initialize chat model(LLM)
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None
    ##initialize chat model
    llm=ChatCohere(cohere_api_key=api_key,
               model=model_name, 
               temperature=0.7,
               streaming=True)  
    ## create prompt template
    prompt = ChatPromptTemplate.from_messages ([
        ("system", "You are a helpful assistant powered by Cohere that provides accurate answers based on the provided context."),
        ("user",  "{question}")
    ])

    ##create chain
    chain = prompt | llm | StrOutputParser()

    return chain

##get chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Cohere API key in the sidebar to use the chatbot.")
    st.markdown("You can get your API key from [Cohere Dashboard](https://dashboard.cohere.com/api-keys).")
    
else:
    ##Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"]) 

    ##Chat input
    if question := st.chat_input("Ask me anything"):
        ##add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question) 

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                ##get response from chain
                for chunk in chain.stream({"question":question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")  # Display loading indicator
                message_placeholder.markdown(full_response)  # Final response without loading indicator

                ## add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                message_placeholder.markdown("Error: " + str(e))

## Examples
st.markdown("---")
st.markdown("### try these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain the concept of embeddings.")
with col2:
    st.markdown("- How does a transformer model work?")
    st.markdown("- What are the applications of LLMs?")
with col3:
    st.markdown("- Describe the attention mechanism.")
    st.markdown("- What is the difference between supervised and unsupervised learning?")

###footer
st.markdown("---")
st.markdown("Developed by [Kartik Awasthi] | [GitHub](https://github.com/VeNOM-ops7)")

