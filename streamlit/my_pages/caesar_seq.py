import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


def let2num(let):
    return ord(let.lower()) - ord('a')


def num2let(num):
    return chr(num + ord('a'))


def caesar(pt:str,key:int):

    if pt.isalpha() and 0 <= key <= 25:
        pt = pt.lower()
        ct = ""
        for i in range(len(pt)):
            ct_num = (let2num(pt[i]) + key) % 26
            ct += num2let(ct_num)
        
        return ct
            

        



def caeser_seq_card(button_callback):
    
    st.write("## :blue[CAESAR CIPHER] (using Seq2Seq models)")


    st.button('Back', on_click=button_callback)

    pt = st.text_input('Enter Plaintext: ')
    key = st.number_input(label = "Enter Key here",max_value=25)

    if pt and key:
        ct = caesar(pt,key)

        st.write(f':blue[Neural Network Output:] {ct}')
        st.write(f':red[Cipher Output:] {ct}')

    st.image('./media/caesar_seq_acc.png')
    st.image('./media/caesar_seq_loss.png')
