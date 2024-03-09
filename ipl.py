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
    st.write("Welcome to our cutting-edge IPL prediction platform, where data meets excitement! Our website is your go-to destination for unraveling the mysteries of IPL match outcomes. By harnessing the power of machine learning, we dissect current match statistics to provide you with a predictive edge like never before. Whether you're a seasoned cricket aficionado or a casual fan, our platform offers a unique blend of real-time data and analytical prowess. Dive into the heart of the game, explore team dynamics, player performance, and strategic insights, all meticulously analyzed by our advanced ML model. Empower your cricket predictions with the precision of data, and elevate your IPL experience with our unparalleled predictive prowess. It's not just about the game; it's about making informed choices and riding the thrill of anticipation. Join us in transforming the way you engage with IPL matches.")
    st.divider()
    st.caption("Made by ------")
if submit:

    crr=(target-runs_left)/((120-balls_left)/6)
    rrr=(runs_left)/((balls_left)/6)
    data=np.array([[batting_team,bowling_team,city,runs_left,balls_left,wickets_left,target,crr,rrr]])

    pred=pipe.predict_proba(data)



    if pred[0,0]>pred[0,1]:
        st.header(str(math.floor(((pred[0,0])*100))) + "%")
        st.success("{} win probability 🥳".format(bowling_team))


        st.header(str(math.floor(((pred[0,1])*100))) + "%")
        st.warning("{} win probability 😭".format(batting_team))



    elif pred[0,0]<pred[0,1]:
        st.header(str(math.floor(((pred[0,0])*100))) + "%")
        st.warning("{} win probability 😭".format(bowling_team))


        st.header(str(math.floor(((pred[0,1])*100))) + "%")
        st.success("{} win probability 🥳".format(batting_team))


    else:
        st.header(str(math.floor(((pred[0,0])*100))) + "%")
        st.success("{} win probability".format(bowling_team))


        st.header(str(math.floor(((pred[0,1])*100))) + "%")
        st.success("{} win probability".format(batting_team))
