import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

teams = {
    "Mumbai Indians": "MI",
    "Sunrisers Hyderabad": "SRH",
    "Chennai Super Kings": "CSK",
    "Punjab Kings": "PBKS",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Rajasthan Royals": "RR",
    "Royal Challengers Bangalore": "RCB"
}

team_colors = {
    "MI": "#045093",
    "SRH": "#f26522",
    "CSK": "#f8cd0a",
    "PBKS": "#d11a2a",
    "KKR": "#3b215d",
    "DC": "#17449b",
    "RR": "#ea1a8c",
    "RCB": "#da1818"
}

cities = [
    "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
    "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
    "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
    "Raipur", "Kimberley", "East London", "Bloemfontein"
]

def show_team_logo(abbr, width=100):
    path = f"team_logos/{abbr}.png"
    if os.path.exists(path):
        st.image(path, width=width)

def plot_pie_chart(pred, labels):
    colors = [team_colors[teams[t]] for t in labels]
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)

def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

def match_summary(bat, bowl, runs_left, balls_left, wickets_left, crr, rrr):
    return (f"{bat} need {runs_left} runs from {balls_left} balls with {wickets_left} wickets in hand.\n"
            f"Current Run Rate: {crr}, Required Run Rate: {rrr}.")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #da1a1a;'>🏏 IPL Win Predictor</h1>
        <p style='font-size:18px;'>Machine learning powered second innings predictor for thrilling IPL finishes</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", list(teams.keys()))
    city = st.selectbox("Match City", cities)
    runs_left = st.number_input("Runs Left", min_value=1)
    wickets_left = st.selectbox("Wickets Left", list(range(1, 11)))

with col2:
    bowling_team = st.selectbox("Bowling Team", list(teams.keys())[::-1])
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
    target = st.number_input("Target Score", min_value=1)

if st.button("🎯 Predict"):
    if runs_left >= target:
        st.error("Runs left must be less than the target score")
    else:
        crr, rrr = calculate_crr_rrr(runs_left, balls_left, target)
        input_data = np.array([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])

        try:
            prediction = pipe.predict_proba(input_data)
            prob_bowl = prediction[0][0]
            prob_bat = prediction[0][1]

            st.markdown("""<hr><h3 style='text-align:center;'>🏆 Match Result Prediction</h3>""", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                show_team_logo(teams[batting_team])
                st.markdown(f"<h4 style='color:{team_colors[teams[batting_team]]};text-align:center'>{batting_team}</h4>", unsafe_allow_html=True)
            with c2:
                show_team_logo(teams[bowling_team])
                st.markdown(f"<h4 style='color:{team_colors[teams[bowling_team]]};text-align:center'>{bowling_team}</h4>", unsafe_allow_html=True)

            plot_pie_chart([prob_bowl, prob_bat], [bowling_team, batting_team])
            st.success(f"{batting_team} chance: {round(prob_bat*100)}%")
            st.warning(f"{bowling_team} chance: {round(prob_bowl*100)}%")

            st.markdown("<hr><h4>📈 Match Stats</h4>", unsafe_allow_html=True)
            stats1, stats2, stats3, stats4 = st.columns(4)
            stats1.metric("Target", target)
            stats2.metric("Runs Left", runs_left)
            stats3.metric("CRR", crr)
            stats4.metric("RRR", rrr)

            st.markdown("<hr><h4>📖 Match Summary</h4>", unsafe_allow_html=True)
            st.info(match_summary(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr))

        except Exception as e:
            st.error(f"Prediction error: {e}")
