import io

import requests
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


st.write("""# stock proce app""")

# get symbol list based on market
url = "https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
response = requests.get(url).content
stocks = pd.read_csv(io.StringIO(response.decode('utf-8')))
stocksInfo = stocks['Symbol'].tolist()

selected_stocks = st.selectbox("Select stock for predication", stocksInfo)


def getTimeRangeAsString(timeRange):
    now = datetime.now()  # current date and time
    return now.strftime("%Y-%m-%d"), (now - timedelta(days=timeRange)).strftime("%Y-%m-%d")

def load_data(ticker):
    data =  yf.download(ticker, startDate, endDate)
    data.reset_index(inplace=True)
    return data



