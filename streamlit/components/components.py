import streamlit as st

def st_card(name, button_callback, button_args):
    key_name='key '+name
    st.markdown(
        f"""
        <div>
            <h3>{name}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    button=st.button('Expand', key=key_name, on_click=button_callback, args=button_args)
    
    