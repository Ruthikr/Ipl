import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Caching model
@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# Team info (colors, logo paths)
teams = {
    "Mumbai Indians":      {"color": "#045093", "logo": "team_logos/MI.png"},
    "Sunrisers Hyderabad": {"color": "#f26522", "logo": "team_logos/SRH.png"},
    "Chennai Super Kings": {"color": "#f8cd0a", "logo": "team_logos/CSK.png"},
    "Punjab Kings":        {"color": "#d11a2a", "logo": "team_logos/PBKS.png"},
    "Kolkata Knight Riders":{"color": "#3b215d", "logo": "team_logos/KKR.png"},
    "Delhi Capitals":      {"color": "#17449b", "logo": "team_logos/DC.png"},
    "Rajasthan Royals":    {"color": "#ea1a8c", "logo": "team_logos/RR.png"},
    "Royal Challengers Bangalore": {"color": "#da1818", "logo": "team_logos/RCB.png"},
}

cities = [
    "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Chandigarh",
    "Pune", "Durban", "Visakhapatnam", "Centurion", "Ahmedabad", "Indore", "Dharamsala",
    "Johannesburg", "Cuttack", "Ranchi", "Port Elizabeth", "Cape Town", "Abu Dhabi", "Sharjah",
    "Raipur", "Kimberley", "East London", "Bloemfontein"
]

# Page config
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Win Predictor")
st.markdown("Get real-time win probabilities using a machine learning model trained on IPL match data.")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Welcome 🤗")
    st.write("Use the inputs below to simulate an ongoing match scenario and predict the winning probability.")
    st.caption("Made by: Ruthik")

# Team selector with badge
def team_badge(team_name):
    logo_path = teams[team_name]["logo"]
    color = teams[team_name]["color"]
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; background: {color}; padding: 8px; border-radius: 10px;">
            <img src="data:image/png;base64,{get_image_base64(logo_path)}" width="30"/>
            <span style="color: white; font-weight: bold;">{team_name}</span>
        </div>
    """, unsafe_allow_html=True)

# Encode logo to base64 for inline image
import base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Input layout
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

# Utility functions
def calculate_crr_rrr(runs_left, balls_left, target):
    overs_bowled = (120 - balls_left) / 6
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")
    return round(crr, 2), round(rrr, 2)

def plot_pie_chart(pred, labels):
    colors = [teams.get(team, {}).get("color", "#cccccc") for team in labels]
    fig, ax = plt.subplots()
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)

def generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr):
    tone = "tense" if rrr > crr else "comfortable"
    if tone == "tense":
        return f"{batting_team} are under pressure! With {runs_left} runs needed off {balls_left} balls and only {wickets_left} wickets in hand, they need {rrr} RPO against a strong {bowling_team} attack."
    else:
        return f"{batting_team} seem in control! They need {runs_left} runs off {balls_left} balls with {wickets_left} wickets remaining. Current run rate is {crr}, and required is {rrr}."

# Prediction block
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

            st.subheader("🏆 Win Probability")
            plot_pie_chart([win_prob_bowling, win_prob_batting], [bowling_team, batting_team])
            st.progress(int(win_prob_batting * 100))

            if win_prob_batting > win_prob_bowling:
                st.success(f"{batting_team} are more likely to win! 🥳 ({round(win_prob_batting*100)}%)")
                st.warning(f"{bowling_team} chance: {round(win_prob_bowling*100)}%")
            else:
                st.success(f"{bowling_team} are more likely to win! 🥳 ({round(win_prob_bowling*100)}%)")
                st.warning(f"{batting_team} chance: {round(win_prob_batting*100)}%)")

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

            st.subheader("📊 Match Situation")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Target", target)
            m2.metric("Runs Left", runs_left)
            m3.metric("Balls Left", balls_left)
            m4.metric("Wickets Left", wickets_left)
            st.metric("Current Run Rate (CRR)", crr)
            st.metric("Required Run Rate (RRR)", rrr)

            st.subheader("📈 Match Progress")
            progress_pct = int(((target - runs_left) / target) * 100)
            st.progress(progress_pct)

            st.subheader("📖 Match Summary")
            story = generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr)
            st.info(story)

            st.divider()

            st.subheader("👕 Team Badges")
            col_a, col_b = st.columns(2)
            with col_a:
                team_badge(batting_team)
            with col_b:
                team_badge(bowling_team)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
