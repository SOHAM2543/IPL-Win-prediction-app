import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page title and styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
        }
        .required-rate {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #FF5733;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">IPL Win Predictor</p>', unsafe_allow_html=True)

# Teams and cities data
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Load model
pipe = pickle.load(open('pipe.pkl','rb'))

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Check if teams are the same
if batting_team == bowling_team:
    st.warning("Please select different teams for batting and bowling.")
    st.stop()

# City selection
selected_city = st.selectbox('Select host city', sorted(cities))

# Historical statistics data (sample dataset)
match_data = {
    'teams': ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals'],
    'matches': [40, 40, 40, 40, 40, 40, 40, 40],
    'wins': [20, 25, 15, 18, 10, 28, 12, 16]
}

history_df = pd.DataFrame(match_data)
team1_stats = history_df[history_df['teams'] == batting_team].iloc[0]
team2_stats = history_df[history_df['teams'] == bowling_team].iloc[0]

team1_win_percent = round((team1_stats['wins'] / team1_stats['matches']) * 100, 2)
team2_win_percent = round((team2_stats['wins'] / team2_stats['matches']) * 100, 2)

# Display historical statistics in table
st.subheader('Historical Statistics')
styled_df = pd.DataFrame({
    'Team': [batting_team, bowling_team],
    'Total Matches': [team1_stats['matches'], team2_stats['matches']],
    'Wins': [team1_stats['wins'], team2_stats['wins']],
    'Win %': [team1_win_percent, team2_win_percent]
})
st.dataframe(styled_df.style.background_gradient(cmap='Blues'))

# Target input
target = st.number_input('Target', step=1)

# Current match stats
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.6, step=0.1, format="%.1f")
with col5:
    wickets = st.selectbox('Wickets out', options=list(range(0, 11)), index=0)  # Dropdown for wickets from 0 to 10

# Prediction
if st.button('Predict Probability'):
    st.write("Prediction in progress...")
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left

    # Prepare input for model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probabilities
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display results
    st.header(batting_team + " - " + str(round(win * 100)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100)) + "%")

    # Bar chart for predictions
    st.bar_chart(pd.DataFrame({'Probability': [win * 100, loss * 100]}, index=[batting_team, bowling_team]))

# Improved run progress graph for 20 overs
df = pd.DataFrame({
    'overs': list(range(21)),
    'target': [target] * 21,
    'score': [0] + [score / overs * (i + 1) for i in range(1, 21)]
})

fig, ax = plt.subplots()
ax.plot(df['overs'], df['target'], color='red', label='Target')
ax.plot(df['overs'], df['score'], color='green', label='Projected Score')

# Adding point markers for projected score
ax.scatter(df['overs'], df['score'], color='green', zorder=5)

# Adding blue point for current score and overs
ax.scatter(overs, score, color='blue', zorder=6, label='Current Score')

ax.fill_between(df['overs'], 0, df['target'], color='red', alpha=0.1)
ax.fill_between(df['overs'], 0, df['score'], color='green', alpha=0.3)
ax.set_xlabel('Overs')
ax.set_ylabel('Score')
ax.legend()
st.pyplot(fig)

# Display required run rate and current run rate
rrr = (target - score) * 6 / (120 - (overs * 6)) if overs < 20 else 0
st.markdown(f'<p class="required-rate">Required Run Rate: {round(rrr, 2)} runs per over</p>', unsafe_allow_html=True)
st.markdown(f'<p class="required-rate">Current Run Rate (CRR): {round(crr, 2)} runs per over</p>', unsafe_allow_html=True)
