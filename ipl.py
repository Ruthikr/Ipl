import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# Teams & Colors
tems = {
    "Mumbai Indians": ("MI", "#045093"),
    "Sunrisers Hyderabad": ("SRH", "#f26522"),
    "Chennai Super Kings": ("CSK", "#f8cd0a"),
    "Punjab Kings": ("PBKS", "#d11a2a"),
    "Kolkata Knight Riders": ("KKR", "#3b215d"),
    "Delhi Capitals": ("DC", "#17449b"),
    "Rajasthan Royals": ("RR", "#ea1a8c"),
    "Royal Challengers Bangalore": ("RCB", "#da1818")
}

cities = ["Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
          "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
          "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
          "Raipur", "Kimberley", "East London", "Bloemfontein"]

# Style
st.markdown("""
    <style>
    .team-card {
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin: 10px auto;
        color: white;
        width: 100%;
    }
    .win-box {
        font-size: 24px;
        padding: 12px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .credit {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        opacity: 0.6;
    }
    </style>
""", unsafe_allow_html=True)

# Utility

def team_logo(team):
    abbr = tems[team][0]
    path = f"team_logos/{abbr}.png"
    if os.path.exists(path):
        return f"<img src='team_logos/{abbr}.png' width='90'>"
    return team

def plot_pie(pred, labels):
    fig, ax = plt.subplots()
    colors = [tems[t][1] for t in labels]
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

def summary(bat, bowl, runs, balls, wkts, crr, rrr):
    if rrr > crr:
        return f"{bat} need {runs} from {balls} balls with {wkts} wickets left. Required rate {rrr} is more than CRR {crr}. Pressure!"
    else:
        return f"{bat} need {runs} from {balls} balls with {wkts} wickets left. CRR is {crr}, and RRR is {rrr}. They look comfortable."

# Header
st.markdown("""
    <h1 style='text-align:center;'>🏏 IPL Win Predictor</h1>
    <p style='text-align:center;'>Powered by machine learning. Predict live match outcomes.</p>
""", unsafe_allow_html=True)

# Inputs
batting_team = st.selectbox("Batting Team", list(tems.keys()))
bowling_team = st.selectbox("Bowling Team", list(tems.keys())[::-1])
city = st.selectbox("City", cities)
runs_left = st.number_input("Runs Left", min_value=1)
balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
wickets_left = st.selectbox("Wickets Left", list(range(1, 11)))
target = st.number_input("Target", min_value=1)

if st.button("Predict 🔮"):
    crr, rrr = calculate_crr_rrr(runs_left, balls_left, target)
    x = np.array([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])
    try:
        proba = pipe.predict_proba(x)
        win_bowling = proba[0][0]
        win_batting = proba[0][1]

        # Team Cards
        st.markdown(f"""
        <div class='team-card' style='background-color:{tems[batting_team][1]}'>
            {team_logo(batting_team)}<br>{batting_team}
        </div>
        <div style='text-align:center; font-weight:bold;'>vs</div>
        <div class='team-card' style='background-color:{tems[bowling_team][1]}'>
            {team_logo(bowling_team)}<br>{bowling_team}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Win Probability")
        plot_pie([win_bowling, win_batting], [bowling_team, batting_team])

        # Win Boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='win-box' style='background-color:{tems[batting_team][1]}'>{batting_team}: {round(win_batting*100)}%</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='win-box' style='background-color:{tems[bowling_team][1]}'>{bowling_team}: {round(win_bowling*100)}%</div>", unsafe_allow_html=True)

        # Match Summary
        st.subheader("Summary")
        st.info(summary(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr))

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target", target)
        col2.metric("Runs Left", runs_left)
        col3.metric("Balls Left", balls_left)
        col4.metric("Wickets", wickets_left)
        st.metric("CRR", crr)
        st.metric("RRR", rrr)

        # Credit
        st.markdown("<div class='credit'>Made with ❤️ by <b>Ruthik</b></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
