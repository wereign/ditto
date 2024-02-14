import streamlit as st

def function():
    a='this is smthing else lol'
    return a

def aes_card(button_callback):
    number=function()
    st.button('Back',on_click=button_callback)
    
    st.markdown(
        f"""
        number obtained is {number}
        """
    )