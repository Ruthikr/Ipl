import streamlit as st
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Streamlit Config
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# Load model (with cache)
@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# Teams, Abbreviations, Colors
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
    "Mumbai Indians": "#045093",
    "Sunrisers Hyderabad": "#f26522",
    "Chennai Super Kings": "#f8cd0a",
    "Punjab Kings": "#d11a2a",
    "Kolkata Knight Riders": "#3b215d",
    "Delhi Capitals": "#17449b",
    "Rajasthan Royals": "#ea1a8c",
    "Royal Challengers Bangalore": "#da1818"
}

cities = [
    "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
    "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
    "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
    "Raipur", "Kimberley", "East London", "Bloemfontein"
]

# Helper: Logo Renderer
def show_team_badge(team, width=100):
    abbr = teams[team]
    logo_path = f"team_logos/{abbr}.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=width)
    else:
        st.markdown(f"🛑 Logo missing for {team}")

# Pie Chart
def plot_pie_chart(pred, labels):
    colors = [team_colors.get(team, "#cccccc") for team in labels]
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)

# Match Summary Text
def generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr):
    tone = "tense" if rrr > crr else "comfortable"
    if tone == "tense":
        return f"{batting_team} are under pressure! With {runs_left} runs needed off {balls_left} balls and only {wickets_left} wickets in hand, they need {rrr} RPO against a strong {bowling_team} attack."
    else:
        return f"{batting_team} seem in control! They need {runs_left} runs off {balls_left} balls with {wickets_left} wickets remaining. Current run rate is {crr}, and required is {rrr}."

# CRR & RRR Calculator
def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

# UI
st.title("🏏 IPL Win Predictor")
st.markdown("Get real-time win probabilities using a machine learning model trained on IPL match data.")
st.divider()

with st.sidebar:
    st.header("Welcome 🤗")
    st.write("Use the inputs below to simulate an ongoing match scenario and predict the winning probability.")
    st.caption("Made by: Ruthik")

# Match Inputs
st.header("Second Innings Predictor")
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team", list(teams.keys()))
    city = st.selectbox("Select City of Playing", cities)
    runs_left = st.number_input("Runs Left to Win", min_value=1)
    wickets_left = st.selectbox("Wickets Remaining", list(range(1, 11)))

with col2:
    bowling_team = st.selectbox("Select Bowling Team", list(teams.keys())[::-1])
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
    target = st.number_input("Target Score", min_value=1)

submit = st.button("🎯 Predict")

# Prediction
if submit:
    if runs_left >= target:
        st.error("Runs left must be less than the target.")
    elif balls_left <= 0:
        st.error("Balls left must be greater than 0.")
    else:
        crr, rrr = calculate_crr_rrr(runs_left, balls_left, target)
        input_data = np.array([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]])

        try:
            prediction = pipe.predict_proba(input_data)
            win_prob_bowling = prediction[0][0]
            win_prob_batting = prediction[0][1]

            # Section 1: Logos & Badges
            st.subheader("🏁 Teams")
            t1, t2 = st.columns(2)
            with t1:
                show_team_badge(batting_team)
                st.markdown(f"<div style='text-align:center;font-size:20px;color:{team_colors[batting_team]}'><b>{batting_team}</b></div>", unsafe_allow_html=True)
            with t2:
                show_team_badge(bowling_team)
                st.markdown(f"<div style='text-align:center;font-size:20px;color:{team_colors[bowling_team]}'><b>{bowling_team}</b></div>", unsafe_allow_html=True)

            # Section 2: Win Pie Chart
            st.subheader("🔮 Win Probability")
            plot_pie_chart([win_prob_bowling, win_prob_batting], [bowling_team, batting_team])
            st.progress(int(win_prob_batting * 100))

            if win_prob_batting > win_prob_bowling:
                st.success(f"{batting_team} are more likely to win! 🥳 ({round(win_prob_batting*100)}%)")
                st.warning(f"{bowling_team} chance: {round(win_prob_bowling*100)}%")
            else:
                st.success(f"{bowling_team} are more likely to win! 🥳 ({round(win_prob_bowling*100)}%)")
                st.warning(f"{batting_team} chance: {round(win_prob_batting*100)}%)")

            # Section 3: Insight Text
            st.markdown("### Verdict Insight")
            diff = abs(win_prob_batting - 0.5)
            if win_prob_batting > 0.8:
                st.success("🎯 Almost there! Batting team in full control.")
            elif win_prob_batting < 0.2:
                st.error("🛑 Very tough! Bowling team has the upper hand.")
            elif diff < 0.1:
                st.info("⚖️ It's a nail-biter! Could go either way.")
            else:
                st.info("Match evenly poised. Keep watching!")

            st.divider()

            # Section 4: Match Stats
            st.subheader("📊 Match Situation")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target", target)
            c2.metric("Runs Left", runs_left)
            c3.metric("Balls Left", balls_left)
            c4.metric("Wickets Left", wickets_left)
            st.metric("Current Run Rate (CRR)", crr)
            st.metric("Required Run Rate (RRR)", rrr)

            # Section 5: Match Progress
            st.subheader("📈 Match Progress")
            progress_pct = int(((target - runs_left) / target) * 100)
            st.progress(progress_pct)

            # Section 6: Summary
            st.subheader("📖 Match Summary")
            story = generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr)
            st.info(story)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
