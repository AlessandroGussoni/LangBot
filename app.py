import os 

import pinecone
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import streamlit as st
from streamlit_chat import message



def get_index(index_name):
    
    pinecone.init(api_key=os.environ['pinecone_key'],
                  environment=os.environ['pinecone_env'])
    index = Pinecone.from_existing_index(index_name=index_name, embedding=OpenAIEmbeddings())
    return index

def main():

    output = None

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    st.set_page_config(page_title="LangBot", layout="wide")
    st.title("Welcome to LangBot 	⛓️")
    st.subheader("""LangBot is trained on LangChain Github code! It can assist you in debugging, learning and understanding LangChain codebase""")
    
    index_name = 'langchain'

    st.sidebar.title("Made by Gustar8")
    
    try:
        db = get_index(index_name=index_name)
        llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm=llm, return_source_documents=True, retriever=db.as_retriever())

        query = st.text_input("Ask me anything about Langchain")

        if query:
            print(st.session_state['chat_history'])
            
            with st.spinner("Generating response..."):
                output = qa({"question": query, "chat_history": st.session_state['chat_history']})

                st.session_state.past.append(query)
                st.session_state.generated.append(output["answer"])
                st.session_state.chat_history.append((query, output['answer']))

        if st.session_state['generated']:

            for i in range(len(st.session_state['generated']) -1, -1, -1):
                sources = "\nSources: \n" + " ".join([source.metadata['source'] for source in output['source_documents']])
                message(st.session_state["generated"][i] + sources, 
                        avatar_style="bottts-neutral",
                        key=str(i))
                message(st.session_state['past'][i], 
                        avatar_style="adventurer-neutral",
                        is_user=True, key=str(i) + '_user')
                
        

    except Exception as e:
        print(e)
        st.write("Whoooops it seems we have an issue in loading our index")


if __name__ == "__main__":
    main()