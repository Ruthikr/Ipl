import streamlit as st
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

# Load model
pipe = pickle.load(open("pipe.pkl", "rb"))

# Static Data
teams = [
    "Mumbai Indians", "Sunrisers Hyderabad", "Chennai Super Kings", "Punjab Kings",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals", "Royal Challengers Bangalore"
]

cities = [
    "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
    "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
    "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
    "Raipur", "Kimberley", "East London", "Bloemfontein"
]

# Streamlit UI
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
st.title("IPL Win Predictor")
st.markdown("Get real-time win probabilities using a machine learning model trained on IPL match data.")
st.divider()

with st.sidebar:
    st.header("Welcome 🤗")
    st.write("Use the inputs below to simulate an ongoing match scenario and predict the winning probability.")
    st.caption("Made by: Ruthik")

st.header("Second Innings Predictor")
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team", teams)
    city = st.selectbox("Select City of Playing", cities)
    runs_left = st.number_input("Runs Left to Win", min_value=1)
    wickets_left = st.selectbox("Wickets Remaining", list(range(1, 11)))

with col2:
    bowling_team = st.selectbox("Select Bowling Team", teams[::-1])
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
    target = st.number_input("Target Score", min_value=1)

submit = st.button("Predict")

def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

def plot_pie_chart(pred, labels):
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=["#0066cc", "#ff9933"])
    ax.axis("equal")
    st.pyplot(fig)

def generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr):
    tone = "tense" if rrr > crr else "comfortable"
    if tone == "tense":
        return f"{batting_team} are under pressure! With {runs_left} runs needed off {balls_left} balls and only {wickets_left} wickets in hand, they need {rrr} RPO against a strong {bowling_team} attack."
    else:
        return f"{batting_team} seem in control! They need {runs_left} runs off {balls_left} balls with {wickets_left} wickets remaining. Current run rate is {crr}, and required is {rrr}."

if submit:
    crr, rrr = calculate_crr_rrr(runs_left, balls_left, target)
    input_data = np.array([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])

    try:
        prediction = pipe.predict_proba(input_data)
        win_prob_bowling = prediction[0][0]
        win_prob_batting = prediction[0][1]

        st.subheader("Win Probability")
        plot_pie_chart([win_prob_bowling, win_prob_batting], [bowling_team, batting_team])

        if win_prob_batting > win_prob_bowling:
            st.success(f"{batting_team} are more likely to win! 🥳 ({round(win_prob_batting*100)}%)")
            st.warning(f"{bowling_team} chance: {round(win_prob_bowling*100)}%")
        else:
            st.success(f"{bowling_team} are more likely to win! 🥳 ({round(win_prob_bowling*100)}%)")
            st.warning(f"{batting_team} chance: {round(win_prob_batting*100)}%")

        st.divider()
        st.subheader("Match Summary")
        story = generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr)
        st.info(story)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
