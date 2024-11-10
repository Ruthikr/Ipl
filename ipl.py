import streamlit as st
import pickle
import numpy as np
import math
import time
import sklearn

pipe=pickle.load(open("pipe.pkl","rb"))
cities=[
"Mumbai",           
"Kolkata",           
"Delhi",             
"Bangalore",         
"Hyderabad",         
"Chennai",           
"Jaipur",             
"Chandigarh",         
"Pune",              
"Durban",            
"Bangalore",          
"Visakhapatnam",      
"Centurion",          
"Ahmedabad",                                
"Indore",              
"Dharamsala",          
"Johannesburg",        
"Cuttack",             
"Ranchi",              
"Port Elizabeth",      
"Cape Town",           
"Abu Dhabi",           
"Sharjah",            
"Raipur",                                   
"Kimberley",           
"East London",         
"Bloemfontein"
]





teams=[
"Mumbai Indians",
"Sunrisers Hyderabad",
"Chennai Super Kings",
"Punjab Kings",
"Kolkata Knight Riders",
"Delhi Capitals",
"Rajasthan Royals",
"Royal Challengers Bangalore",

    
]


st.set_page_config(
    page_title="Ipl win Predictor ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="auto",
    )
st.title("IPL Win Predictor")
st.divider()
st.header("Second Innings")



batting_team=st.selectbox("Select Batting Team",teams)

bowling_team=st.selectbox("Select Bowling Team",teams[::-1])

city=st.selectbox("Select City of playing",cities)

runs_left=st.number_input("enter number of runs left to win",value=1)

balls_left=st.number_input("enter number of balls left",value=1)

wickets_left=st.selectbox("number of wickets remaining",list(range(1,11)))

target=st.number_input("Target score",value=1)

submit=st.button("predict")
with st.sidebar:
    st.header("Welcome 🤗")
    st.write("""IPL Win Predictor
Welcome to the IPL Win Predictor, a machine learning-powered tool designed to predict the probability of a team winning an ongoing IPL match.

How It Works:

Input Match Details: Enter key match details such as teams playing, toss winner, and match venue.

Prediction: Based on historical IPL match data from the first season to the present, the model uses Logistic Regression to calculate the likelihood of each team winning, providing you with a balanced probability.

Interactive Dashboard: View real-time predictions and adjust input data to see how different match conditions influence the outcome.


Key Features:

Data-driven predictions based on years of IPL data.

Simple and intuitive interface for easy match input.

Real-time team win probability calculation using Logistic Regression.


Explore the data-driven insights and make your next IPL match predictions! """)
    st.divider()
    st.caption("Made by- Ruthik")
if submit:

    crr=(target-runs_left)/((120-balls_left)+(0.0000000000000000001)/6)
    rrr=(runs_left)/((balls_left)/6)
    data=np.array([[batting_team,bowling_team,city,runs_left,balls_left,wickets_left,target,crr,rrr]])

    pred=pipe.predict_proba(data)



    if pred[0,0]>pred[0,1]:
        st.header(str(round(((pred[0,0])*100))) + "%")
        st.success("{} win probability 🥳".format(bowling_team))


        st.header(str(round(((pred[0,1])*100))) + "%")
        st.warning("{} win probability 😭".format(batting_team))



    elif pred[0,0]<pred[0,1]:
        st.header(str(round(((pred[0,0])*100))) + "%")
        st.warning("{} win probability 😭".format(bowling_team))


        st.header(str(round(((pred[0,1])*100))) + "%")
        st.success("{} win probability 🥳".format(batting_team))


    else:
        st.header(str(round(((pred[0,0])*100))) + "%")
        st.success("{} win probability".format(bowling_team))


        st.header(str(round(((pred[0,1])*100))) + "%")
        st.success("{} win probability".format(batting_team))
