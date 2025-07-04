import streamlit as st  
import pickle  
import numpy as np  
import matplotlib.pyplot as plt  
  
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")  
  
# Load ML model  
@st.cache_resource  
def load_model():  
    return pickle.load(open("pipe.pkl", "rb"))  
  
pipe = load_model()  
  
# Team info  
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
  
# Custom CSS  
st.markdown("""  
    <style>  
    .card {  
        border-radius: 12px;  
        padding: 20px;  
        text-align: center;  
        font-weight: bold;  
        font-size: 18px;  
        background-color: #f5f5f5;  
        margin-bottom: 10px;  
    }  
    </style>  
""", unsafe_allow_html=True)  
  
# Functions  
def plot_pie_chart(pred, labels):  
    colors = [teams[l][1] for l in labels]  
    fig, ax = plt.subplots()  
    ax.pie(pred, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)  
    ax.axis('equal')  
    st.pyplot(fig)  
  
def generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr):  
    overs_left = balls_left // 6  
    balls_remaining = balls_left % 6  
    time_phrase = f"{overs_left} overs" + (f" and {balls_remaining} balls" if balls_remaining > 0 else "")  
  
    if rrr > crr:  
        return (  
            f"{batting_team} need {runs_left} runs from {time_phrase} with {wickets_left} wickets in hand. "  
            f"The required run rate is {rrr}, which is higher than the current rate of {crr}. "  
            f"{batting_team} must accelerate the scoring or risk falling behind."  
        )  
    else:  
        return (  
            f"{batting_team} need {runs_left} runs from {time_phrase} with {wickets_left} wickets in hand. "  
            f"With a current run rate of {crr} exceeding the required rate of {rrr}, "  
            f"{batting_team} are in a strong position to chase the target."  
        )  
  
def calculate_crr_rrr(runs_left, balls_left, target):  
    overs_bowled = (120 - balls_left) / 6  
    crr = (target - runs_left) / overs_bowled if overs_bowled > 0 else 0  
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float("inf")  
    return round(crr, 2), round(rrr, 2)  
  
# Header  
st.markdown("<h1 style='text-align:center;'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)  
st.markdown("<p style='text-align:center;'>Get real-time win probabilities using a machine learning model trained on IPL match data.</p>", unsafe_allow_html=True)  
st.divider()  
  
# Inputs  
col1, col2 = st.columns(2)  
  
with col1:  
    batting_team = st.selectbox("Batting Team", list(teams.keys()))  
    city = st.selectbox("City", cities)  
    runs_left = st.number_input("Runs Left", min_value=1)  
    wickets_left = st.selectbox("Wickets Remaining", list(range(1, 11)))  
  
with col2:  
    available_bowling_teams = [team for team in teams.keys() if team != batting_team]  
    bowling_team = st.selectbox("Bowling Team", available_bowling_teams)  
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)  
    target = st.number_input("Target Score", min_value=1)  
  
submit = st.button("Predict")  
  
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
  
            # Pie Chart  
            st.subheader("Win Probability")  
            plot_pie_chart([win_prob_bowling, win_prob_batting], [bowling_team, batting_team])  
  
            # Probability Cards  
            prob_cols = st.columns(2)  
            with prob_cols[0]:  
                st.markdown(f"""  
                    <div class='card' style='background-color:{teams[batting_team][1]}; color:white;'>  
                        {batting_team}<br><span style='font-size:28px'>{round(win_prob_batting*100, 1)}%</span><br>Win Probability  
                    </div>  
                """, unsafe_allow_html=True)  
            with prob_cols[1]:  
                st.markdown(f"""  
                    <div class='card' style='background-color:{teams[bowling_team][1]}; color:white;'>  
                        {bowling_team}<br><span style='font-size:28px'>{round(win_prob_bowling*100, 1)}%</span><br>Win Probability  
                    </div>  
                """, unsafe_allow_html=True)  
  
            # Match Stats  
            st.subheader("Match Situation")  
            match_stats = st.columns(4)  
            match_stats[0].metric("Target", target)  
            match_stats[1].metric("Runs Left", runs_left)  
            match_stats[2].metric("Balls Left", balls_left)  
            match_stats[3].metric("Wickets Left", wickets_left)  
            st.metric("CRR", crr)  
            st.metric("RRR", rrr)  
  
            # Summary  
            st.subheader("Match Summary")  
            summary = generate_match_story(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr)  
            st.info(summary)  
  
        except Exception as e:  
            st.error(f"Prediction failed: {e}")
