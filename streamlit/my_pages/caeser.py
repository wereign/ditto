import os
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

def train_model():
    max_val = 26
    mod = 26
    alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Data.
    K = np.random.randint(0, 27, size=(250_000,)).reshape(-1, 1)
    I = np.random.randint(0, max_val, size=(250_000,)).reshape(-1, 1)
    X = np.hstack((I,K))
    Y = (K + I) % 26

    # validation Data.
    Kv = np.random.randint(0, 27, size=(10_000,)).reshape(-1, 1)
    Iv = np.random.randint(0, max_val, size=(10_000,)).reshape(-1, 1)
    Xv = np.hstack((Iv,Kv))
    Yv = (Kv + Iv) % 26

    # Model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1000, 'relu', input_shape=(2,)),
        tf.keras.layers.Dense(26, 'softmax'),
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train.
    history=model.fit(X, Y, batch_size=100, epochs=120, validation_data=(Xv, Yv))
    
    model.save('caeser_stream.h5')
    print('MODEL HAS BEEN SAVED\n')
    
    if os.path.exists('model_history.json'):
        with open('model_history.json','r+') as file:
            data=json.load(file)
            data['caeser_stream']=history.history
            file.seek(0)
            file.truncate(0)
            json.dump(data,file)
            print('FILE EXISTS SO HAVE OVERWRITTEN')
    else:
        with open ('model_history.json','w+') as file:
            data={'caeser_stream':history.history}
            json.dump(data,file)
            print('FILE DOESNT EXIST SO CREATED')
    
    return history

def load_model():
    history=tf.keras.models.load_model('caeser_stream.h5')
    print('LOADING MODEL')
    return history

    
def caeser_card(button_callback):
    
    st.button('Back',on_click=button_callback)

    st.write("## :blue[CAESAR CIPHER] (thin deep neural network)")

    max_val = 26
    
    if os.path.exists('caeser_stream.h5'):
        model=load_model()
    else:
        history=train_model()
        model=history.model
    with open('model_history.json','r') as hist:
        hist=json.load(hist)
        history_metrics=hist['caeser_stream']
        print('HOME STRETCH')

    Iw=st.text_input("Enter plaintext").lower()
    key=int(st.number_input("Enter key"))
    
    if key!='' and len(Iw)!=0:
        
        iw_list=[]
        for count in Iw:
            iw_list.append(ord(count)-97)
        It=np.array(iw_list).reshape(-1,1)
        
        kw_list=[key]
        Kt=np.array(kw_list*len(Iw)).reshape(-1,1)
        
        Xt=np.hstack((It,Kt))
        Yt=(Kt+It)%26    
        print(Yt)
        Y_preds = model.predict(Xt)

        Y_preds = Y_preds.argmax(axis=1)
        Yp=np.array(Y_preds)
        
        input_list=list(map(lambda x: chr(x + ord('a')),It.flatten()))
        
        predicted_list=list(map(lambda x: chr(x + ord('a')),Yp.flatten()))
        target_list=list(map(lambda x: chr(x + ord('a')),Yt.flatten()))
            
        data={'input_character':input_list,'predicted_character':predicted_list,'target_character':target_list}
        
        df=pd.DataFrame(data=data)
        df=df[df['input_character']!=' ']
        st.dataframe(df)
    
        empty=False
    else:
        empty=True
        
    if empty:
        st.markdown('##### Enter values into above test boxes to interact with model')

    st.image('./media/stream_caesar_accuracy.png')
    st.image('./media/stream_caesar_loss.png')
