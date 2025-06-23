import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ----- Config -----
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# ----- Teams, Colors, Logos -----
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

# ----- Custom CSS -----
st.markdown("""
    <style>
    body { background-color: #121212; }
    .block-container { padding-top: 2rem; }
    .team-card {
        padding: 1.5rem; border-radius: 15px; color: white; font-weight: bold;
        text-align: center; margin-bottom: 1rem;
    }
    .stat-box {
        background: #1e1e1e; border-radius: 10px; padding: 10px; text-align: center;
        color: white; font-size: 16px; font-weight: bold;
    }
    footer { text-align: center; color: gray; padding-top: 2rem; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# ----- Utility Functions -----
def show_logo(team, width=100):
    abbr = teams[team][0]
    path = f"team_logos/{abbr}.png"
    if os.path.exists(path):
        st.image(path, width=width)
    else:
        st.write("Logo not found")

def plot_pie(pred, labels):
    colors = [teams[l][1] for l in labels]
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)

def generate_story(bat, bowl, runs, balls, wickets, crr, rrr):
    if rrr > crr:
        return f"{bat} need {runs} runs in {balls} balls with {wickets} wickets. RRR {rrr} > CRR {crr}. Pressure on!"
    else:
        return f"{bat} need {runs} runs in {balls} balls with {wickets} wickets. RRR {rrr} < CRR {crr}. Looking good!"

def calc_crr_rrr(runs, balls, target):
    overs = (120 - balls) / 6
    crr = (target - runs) / overs if overs > 0 else 0
    rrr = runs / (balls / 6) if balls > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

# ----- Header -----
st.markdown("""
    <h1 style='text-align:center; color:white;'>🏏 IPL Win Predictor</h1>
    <p style='text-align:center; color:gray;'>Live win probabilities based on match state</p>
""", unsafe_allow_html=True)
st.divider()

# ----- Inputs -----
col1, col2 = st.columns(2)
with col1:
    bat_team = st.selectbox("Batting Team", list(teams.keys()))
    city = st.selectbox("City", cities)
    runs_left = st.number_input("Runs Left", 1)
    wickets_left = st.selectbox("Wickets Left", list(range(1, 11)))

with col2:
    bowl_team = st.selectbox("Bowling Team", list(teams.keys())[::-1])
    balls_left = st.number_input("Balls Left", 1, 120)
    target = st.number_input("Target", 1)

if st.button("Predict 🔮"):
    if runs_left >= target:
        st.error("Runs left must be less than the target")
    else:
        crr, rrr = calc_crr_rrr(runs_left, balls_left, target)
        inp = np.array([[bat_team, bowl_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])

        try:
            prob = pipe.predict_proba(inp)[0]
            win_bowl, win_bat = prob[0], prob[1]

            team1, team2 = st.columns(2)
            with team1:
                st.markdown(f"""
                    <div class='team-card' style='background:{teams[bat_team][1]}'>
                        {bat_team}<br>
                        <img src='team_logos/{teams[bat_team][0]}.png' width='100'><br>
                        <span style='font-size:24px'>{round(win_bat*100)}%</span> win chance
                    </div>
                """, unsafe_allow_html=True)

            with team2:
                st.markdown(f"""
                    <div class='team-card' style='background:{teams[bowl_team][1]}'>
                        {bowl_team}<br>
                        <img src='team_logos/{teams[bowl_team][0]}.png' width='100'><br>
                        <span style='font-size:24px'>{round(win_bowl*100)}%</span> win chance
                    </div>
                """, unsafe_allow_html=True)

            # Chart
            st.markdown("<h3 style='text-align:center; color:white;'>Win Distribution</h3>", unsafe_allow_html=True)
            plot_pie([win_bowl, win_bat], [bowl_team, bat_team])

            # Stats Row
            st.markdown("<h3 style='color:white;'>Match Stats</h3>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='stat-box'>Target<br>{target}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-box'>Runs Left<br>{runs_left}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-box'>Balls Left<br>{balls_left}</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='stat-box'>Wickets Left<br>{wickets_left}</div>", unsafe_allow_html=True)

            # RRR & CRR
            st.markdown(f"<div class='stat-box'>CRR: {crr} | RRR: {rrr}</div>", unsafe_allow_html=True)

            # Summary
            st.markdown("<h3 style='color:white;'>Match Story</h3>", unsafe_allow_html=True)
            st.info(generate_story(bat_team, bowl_team, runs_left, balls_left, wickets_left, crr, rrr))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----- Footer -----
st.markdown("""
    <footer>
        🛠️ Made by: <b>Ruthik</b> | Built with Streamlit ❤️
    </footer>
""", unsafe_allow_html=True)
