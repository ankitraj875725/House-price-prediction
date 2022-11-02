import streamlit as st
import pandas as pd
import pickle
import numpy as np

pickle_in=open('banglore house price prediction model.pkl','rb')
classifier=pickle.load(pickle_in)

def Welcome():
    return 'WELCOME ALL !'

def predict_price(location,sqft,bath,bhk):
    x=np.zeroes(243)
    x[0]=sqft
    x[1]=bath
    x[2]=bhk

    return np.round(classifier.predict([x])[0],3)
def main():
    home=pd.read_csv("Bengaluru_House_Data.csv")
    loc=home['location'].unique()
    st.title("Bengalore House Rate Prediction")

if __name__=='__main__':
    main()

    
