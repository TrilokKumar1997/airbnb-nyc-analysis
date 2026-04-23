# NYC Airbnb Price Analyzer

An interactive dashboard analyzing 48,616 NYC Airbnb listings with a
machine learning price predictor built with Python and Streamlit.

**Live demo:** YOUR_STREAMLIT_URL

## Features
- Price predictor — enter borough, neighbourhood, room type and get predicted nightly price
- Market analysis by borough, room type and neighbourhood
- Interactive NYC listings map
- SQL-powered analysis using SQLite

## Key findings
- Tribeca is the most expensive neighbourhood at $330/night average
- Manhattan entire homes average $250/night vs $90 in the Bronx
- Neighbourhood and room type are the strongest price predictors

## Tech stack
- Python · pandas · scikit-learn · Plotly · Streamlit · SQLite

## Data source
Kaggle NYC Airbnb Open Data 2019 — 48,616 listings

## How to run locally
git clone https://github.com/TrilokKumar1997/airbnb-nyc-analysis.git
cd airbnb-nyc-analysis
pip install -r requirements.txt
streamlit run app.py

## Author
Trilok Kumar — Data Science MS, University of New Haven
