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
    # val_loss = history.history['val_loss']
    # val_acc = history.history['val_accuracy']
    # acc = history.history['accuracy']
    # loss = history.history['loss']
    # epochs = list(range(120))
    
    model.save('caeser_stream.h5')
    print('MODEL HAS BEEN SAVED\n')
    
    # caeser_stream_dict={'accuracy':acc,'val_accuracy':val_acc,'loss':loss,'val_loss':val_loss,'epochs':epochs}
    
    # try: 
    #     with open('model_history.json','r+') as file:
    #         dictionary=json.load(file)
    #         dictionary['caeser_stream']=caeser_stream_dict
    #         json.dump(dictionary,file)
    # except FileNotFoundError:
    #     with open('model_history.json','w+') as file:
    #         dictionary={'caeser_stream':caeser_stream_dict}
    #         json.dump(dictionary,file)

    # with open('caeser_stream.pkl','+wb') as file:
    #     pickle.dump(history,file)
    
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
    # Test Data
    Kt = np.random.randint(0, 27, size=(10_000,)).reshape(-1, 1)
    It = np.random.randint(0, max_val, size=(10_000,)).reshape(-1, 1)
    Xt = np.hstack((It,Kt))
    Yt = (Kt + It) % 26

    Y_preds = model.predict(Xt)

    Y_preds = Y_preds.argmax(axis=1)
    
    fig,ax=plt.subplots(1,2)
    
    sns.lineplot(ax=ax[0],x=range(len(history_metrics['accuracy'])),y=history_metrics['accuracy'],label='Training')
    sns.lineplot(ax=ax[0],x=range(len(history_metrics['val_accuracy'])),y=history_metrics['val_accuracy'],label='Validation')
    ax[0].legend()
    ax[0].set_title('Accuracy vs Epoch')
    
    sns.lineplot(ax=ax[1],x=range(len(history_metrics['loss'])),y=history_metrics['loss'],label='Training')
    sns.lineplot(ax=ax[1],x=range(len(history_metrics['val_loss'])),y=history_metrics['val_loss'],label='Validation')
    ax[1].legend()
    ax[1].set_title('Loss vs Epoch')
    
    st.pyplot(fig)