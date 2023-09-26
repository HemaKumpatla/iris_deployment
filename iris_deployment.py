#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import numpy as np
import sklearn

with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Iris Flower Classification")
st.markdown("By Hema Nageswari Devi")

st.subheader("Enter Sepal Length, Sepal Width, Petal Width and Petal Length:")
user_input1 = st.number_input('Enter SL',0.00,20.00,step=0.25)
user_input2 = st.number_input('Enter SW',0.00,20.00,step=0.25)
user_input3 = st.number_input('Enter PL',0.00,20.00,step=0.25)
user_input4 = st.number_input('Enter PW',0.00,20.00,step=0.25)

val=np.array([user_input1,user_input2,user_input3,user_input4]).reshape(1,-1)
pred=model.predict(val)

if st.button("Predict"):
    st.success(f"Your predicted Flower Type is {pred[0]}")


# In[ ]:




