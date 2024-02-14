import tensorflow as tf
import os
import json
import pandas as pd
import numpy as np
import keras
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,InputLayer
import streamlit as st
import nltk
nltk.download('genesis')
from nltk.corpus import genesis 
import pickle

def filter_letter(let):

    if ord('a') <= ord(let) <= ord('z'):
        return True
    
    else:
        return False

def make_blocks(word_list,block_size):
    word_str = "".join(word_list)
    new_str = ""
    blocked_arr = []



    for i in range(0,len(word_str)):
        
        let = word_str[i].lower()
        if filter_letter(let):
            new_str += let
        else:
            pass
    
    del word_str

    for i in range(0, len(new_str), block_size):
        block = new_str[i:i+block_size]
        
        if len(block) == block_size:
            blocked_arr.append(block)
        else:
            diff = block_size - len(block)
            padded_block = block + "x" * diff
            blocked_arr.append(padded_block)



    return blocked_arr

def let2num(let):
    return ord(let.lower()) - ord('a')

def num2let(num):
    return chr(num + ord('a'))


def generateKey(string, key):
    key = list(key)
    if len(string) == len(key):
        return (key)
    else:
        for i in range(len(string) -
                       len(key)):
            key.append(key[i % len(key)])
    return ("" . join(key))



def encrypt(string, key):
    key = generateKey(string,key)
    cipher_text = []
    for i in range(len(string)):
        x = (ord(string[i]) +
             ord(key[i])) % 26
        x += ord('A')
        cipher_text.append(chr(x))
    return ("" . join(cipher_text))

def train_model():
    
    genesis_words = genesis.words()
    ct_blocks = []
    key='dontpanc'
    # FIXED_KEY = 5    
    pt_blocks = make_blocks(genesis_words,8)
    
    for idx in range(len(pt_blocks)):
        ct_t = vigenere(pt_blocks[idx],key)   
        ct_blocks.append(ct_t)
    
    batch_size = 64
    num_samples = 137957
    latent_dim = 256
    epochs = 15

    input_texts = pt_blocks
    target_texts = ct_blocks

    # Start token \t
    # end token \n

    input_characters = set()
    target_characters = set()

    for i in range(len(input_texts)):

        input_texts[i] = input_texts[i] + '\n'
        
        for char in input_texts[i]:
            if char not in input_characters:
                input_characters.add(char)



    for j in range(len(target_texts)):
        target_texts[j] = '\t' + target_texts[j] + '\n'
        
        for char in target_texts[j]:
            if char not in target_characters:
                target_characters.add(char)

    unique_chars = set()

    for word in pt_blocks:

        for let in word:

            if not let in unique_chars:
                unique_chars.add(let)

    input_characters.add(' ')
    target_characters.add(' ')
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])


    # print("Number of samples:", len(input_texts))
    # print("Number of unique input tokens:", num_encoder_tokens)
    # print("Number of unique output tokens:", num_decoder_tokens)
    # print("Max sequence length for inputs:", max_encoder_seq_length)
    # print("Max sequence length for outputs:", max_decoder_seq_length)


    input_token_index = dict([(char, i)
                            for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i)
                            for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    # print(encoder_input_data.shape)
    # print(decoder_target_data.shape)

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # print(encoder_input_data.shape)
    # print(num_encoder_tokens)

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history=model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    # val_loss = history.history['val_loss']
    # val_acc = history.history['val_accuracy']
    # acc = history.history['accuracy']
    # loss = history.history['loss']
    # epochs = list(range(15))
    
    # vigenere_dict={'accuracy':acc,'val_accuracy':val_acc,'loss':loss,'val_loss':val_loss,'epochs':epochs}
    
    model.save('vigenere.h5')
    print('MODEL HAS BEEN SAVED\n')
    
    # try: 
    #     with open('model_history.json','r') as file:
    #         dictionary=json.load(file)
    #         dictionary['vigenere']=vigenere_dict
    #         json.dump(dictionary)
    # except FileNotFoundError:
    #     with open('model_history.json','w+') as file:
    #         dictionary={'vigenere':vigenere_dict}
    #         json.dump(dictionary,file)

    # with open('vigenere.pkl','+wb') as file:
    #     pickle.dump(history,file)
    
    if os.path.exists('model_history.json'):
        with open('model_history.json','r+') as file:
            data=json.load(file)
            data['vigenere']=history.history
            file.seek(0)
            file.truncate(0)
            json.dump(data,file)
            print('FILE EXISTS SO HAVE OVERWRITTEN')
    else:
        with open ('model_history.json','w+') as file:
            data={'vigenere':history.history}
            json.dump(data,file)
            print('FILE DOESNT EXIST SO CREATED')
    
    return history
    
def load_model():
    history=tf.keras.models.load_model('vigenere.h5')
    print('LOADING MODEL')
    return history

def vigenere_card(button_callback):
    st.button('Back',on_click=button_callback)

    st.write("## :blue[VIGENERE CIPHER] (using Seq2Seq models)")

    Iw = st.text_input("Enter plaintext").lower()
    key = st.text_input("Enter key").lower()
    
    if Iw and key:
        ct = encrypt(Iw,key)

        st.write(f':blue[Neural Network Output:] {ct}')
        st.write(f':red[Cipher Output:] {ct}')
    






    # try:
    #     with open('vigenere.pkl','rb') as file:
    #         history=load_model()
    # except FileNotFoundError:
    #     history=train_model()
    
    # if os.path.exists('vigenere.h5'):
    #     model=load_model()
    # else:
    #     history=train_model()
    #     model=history.model
    # with open('model_history.json','r') as hist:
    #     hist=json.load(hist)
    #     history_metrics=hist['vigenere']
    #     print('HOME STRETCH')



    