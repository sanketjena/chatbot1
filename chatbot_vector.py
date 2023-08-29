#pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
from dotenv import load_dotenv
import os
# import langchain
# langchain.verbose = False

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


#st.set_page_config(layout="wide")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.squarespace-cdn.com/content/v1/5d2c8662957f82000116912e/1617996155519-FYHLWSXE57QHRSI1O29T/banner1.png");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-size: 2000px 350px;
background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



#CSS for Logo
st.markdown(
        """
        <style>
        .logo-container {
            position: absolute;
            top: -70px; /* Adjust the top position */
            left: 1100px; /* Adjust the left position */
            z-index: 999; /* Ensure the logo appears above other content */
        }
        .logo {
            width: 190px;
            height: 40px;
        }
        </style>
        """
        , unsafe_allow_html=True
    )

# Logo in the corner
st.markdown(
        """
        <div class="logo-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/AbbVie_logo.svg/1280px-AbbVie_logo.svg.png" alt="Logo" class="logo">
        </div>
        """
        , unsafe_allow_html=True
    )


st.markdown("""
        <h1 style="text-align: left; color: #071D49; margin-left: 200px;">DATA TREASURY</h1>
        """, unsafe_allow_html=True)
    
st.markdown('<span style="color: #071D49;font-size: 32px;margin-left: 230px;">Ask Your Question</span>', unsafe_allow_html=True)

st.text("")
st.text("")


tmp_file_path="Gen_AI_Glossary Sample.csv"
loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
data = loader.load()

embeddings = OpenAIEmbeddings()
vectors = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo',verbose=False),
                                                                    retriever=vectors.as_retriever())

def conversational_chat(query):
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about "]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk about your data here (:", key='input')
        submit_button = st.form_submit_button(label='Submit')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            
#streamlit run tuto_chatbot_csv.py