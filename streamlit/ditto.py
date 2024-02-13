import streamlit as st
from components.components import st_card
from pages.caeser import caeser_card
from pages.aes import aes_card
from pages.vigenere import vigenere_card

if 'start' not in st.session_state:
    st.set_page_config(layout='wide',page_title=' Ditto')
    st.session_state['start']=True
    st.session_state['page']='main_page'

def change_page_state(page_name):
    st.session_state.page=page_name
    
def homepage():
    st.session_state.page='main_page'

def display_page():
    
    if st.session_state.page=='main_page':

        model_container=st.container(border=True)
        model_cols=model_container.columns(3)

        with model_cols[0]:
            st_card(name='Vigenere',button_callback=change_page_state,button_args=['des'])

        with model_cols[1]:
            st_card(name='AES',button_callback=change_page_state,button_args=['aes'])

        with model_cols[2]:
            st_card(name='Caeser',button_callback=change_page_state,button_args=['caeser'])

    elif st.session_state.page=='caeser':
        caeser_card(button_callback=homepage)
    elif st.session_state.page=='aes':
        aes_card(button_callback=homepage)
    elif st.session_state.page=='des':
        vigenere_card(button_callback=homepage)
        
display_page()
