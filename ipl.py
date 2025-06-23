import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# Team data
teams = {
    "Mumbai Indians": ("MI", "#045093"),
    "Sunrisers Hyderabad": ("SRH", "#f26522"),
    "Chennai Super Kings": ("CSK", "#f8cd0a"),
    "Punjab Kings": ("PBKS", "#d11a2a"),
    "Kolkata Knight Riders": ("KKR", "#3b215d"),
    "Delhi Capitals": ("DC", "#17449b"),
    "Rajasthan Royals": ("RR", "#ea1a8c"),
    "Royal Challengers Bangalore": ("RCB", "#da1818")
}

cities = [
    "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
    "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
    "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
    "Raipur", "Kimberley", "East London", "Bloemfontein"
]

# Style
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
st.markdown("""
    <style>
        body { background-color: #0d1117; color: white; }
        .team-card {
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
        }
        .win-box {
            border-radius: 12px;
            padding: 20px;
            font-size: 22px;
            text-align: center;
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Functions
def show_logo(team):
    abbr = teams[team][0]
    logo_path = f"team_logos/{abbr}.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.write(f"{team} logo missing")

def plot_pie_chart(pred, labels):
    colors = [teams[l][1] for l in labels]
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float('inf')
    return round(crr, 2), round(rrr, 2)

def generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr):
    if rrr > crr:
        return f"{batting_team} are chasing {runs_left} off {balls_left} with {wickets_left} wickets left. RRR {rrr} > CRR {crr} - tense situation!"
    else:
        return f"{batting_team} are chasing {runs_left} off {balls_left} with {wickets_left} wickets left. RRR {rrr} < CRR {crr} - looking strong!"

# Header
st.markdown("<h1 style='text-align:center;color:white;'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Made with ❤️ by <b>Ruthik</b></p>", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", list(teams.keys()))
    city = st.selectbox("City", cities)
    runs_left = st.number_input("Runs Left", min_value=1)
    wickets_left = st.selectbox("Wickets Left", list(range(1, 11)))

with col2:
    bowling_team = st.selectbox("Bowling Team", list(teams.keys())[::-1])
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
    target = st.number_input("Target Score", min_value=1)

submit = st.button("Predict")

if submit:
    crr, rrr = calculate_crr_rrr(runs_left, balls_left, target)
    input_data = np.array([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])
    try:
        pred = pipe.predict_proba(input_data)
        win_bowling = pred[0][0]
        win_batting = pred[0][1]

        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"<div class='team-card' style='background-color:{teams[batting_team][1]};'>", unsafe_allow_html=True)
            show_logo(batting_team)
            st.markdown(f"{batting_team}</div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f"<div class='team-card' style='background-color:{teams[bowling_team][1]};'>", unsafe_allow_html=True)
            show_logo(bowling_team)
            st.markdown(f"{bowling_team}</div>", unsafe_allow_html=True)

        st.markdown("<h3 style='text-align:center'>Win Probability</h3>", unsafe_allow_html=True)
        plot_pie_chart([win_bowling, win_batting], [bowling_team, batting_team])

        w1, w2 = st.columns(2)
        with w1:
            st.markdown(f"<div class='win-box' style='background-color:{teams[batting_team][1]};'>{batting_team}<br>{round(win_batting*100,1)}%</div>", unsafe_allow_html=True)
        with w2:
            st.markdown(f"<div class='win-box' style='background-color:{teams[bowling_team][1]};'>{bowling_team}<br>{round(win_bowling*100,1)}%</div>", unsafe_allow_html=True)

        st.subheader("Match Situation")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Target", target)
        s2.metric("Runs Left", runs_left)
        s3.metric("Balls Left", balls_left)
        s4.metric("Wickets", wickets_left)
        st.metric("CRR", crr)
        st.metric("RRR", rrr)

        st.subheader("Summary")
        st.info(generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
