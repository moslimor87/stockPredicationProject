import io
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}
</style>
"""


def getTimeRangeAsString(timeRange):
    print("delta:", timeRange)
    now = datetime.now()  # current date and time
    return now.strftime("%Y-%m-%d"), (now - timedelta(days=timeRange)).strftime("%Y-%m-%d")

def getNumberOfDaysForTraining():

    st.radio(
        "Select amount of data for training:",
        ["month", "3 month", "week"],
        key="option",
        label_visibility= 'visible',
        horizontal= True
    )

    futureDaysToPredicate = 7
    days = futureDaysToPredicate

    if st.session_state.option == "week":
        days = days + 7
    if st.session_state.option == "month":
        days = days + 30
    if st.session_state.option == "3 month":
        days = futureDaysToPredicate + 90
    return days


def downloadStockDetails(ticker, timeRangeForPredication):
    print('date range:', timeRangeForPredication)
    return yf.download(ticker, timeRangeForPredication[1], timeRangeForPredication[0])


def calculatePredicateData(data, classifier, futureDaysToPredicate, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    classifier.fit(x_train, y_train)
    print("data:",data['Close'])
    print("data size :", data['Close'].size)
    x_future = data.drop(['Prediction'], 1)[:-futureDaysToPredicate]
    x_future = x_future.tail(futureDaysToPredicate)
    x_future = np.array(x_future)
    print("x future:", x_future)

    tree_prediction = classifier.predict(x_future)
    predictions = tree_prediction
    valid = data[X.shape[0]:]
    print("predivated:", predictions)
    print("valid:", valid)
    valid['Predictions'] = predictions

    valid['Predictions'] = predictions
    return valid['Predictions']


def precidcateBasedOnDecisionTreeClassifier(data):

    futureDaysToPredicate = 7

    data['Prediction'] = data['Close'].shift(-futureDaysToPredicate)
    print("data Prediction:", data['Prediction'])

    X = np.array(data.drop(['Prediction'], 1))[:-futureDaysToPredicate]
    y = np.array(data['Prediction'])[:-futureDaysToPredicate]

    models = {
                'ID3': DecisionTreeClassifier(criterion="entropy", max_depth=len(data['Close'])-1, max_features="sqrt"),
                'CART': DecisionTreeClassifier(criterion="gini", max_depth=len(data['Close'])-1, max_features="sqrt")
            }

    predicatedData = pd.DataFrame()

    for name, classifier in models.items():
        y = y.astype('int')
        predicatedData[name] = calculatePredicateData(data, classifier, futureDaysToPredicate, X,y)

    return predicatedData

def precidcateBasedOnDemandClassifier(data):

        futureDaysToPredicate = 7

        data['Prediction'] = data['Close'].shift(-futureDaysToPredicate)
        print("data Prediction:", data['Prediction'])

        X = np.array(data.drop(['Prediction'], 1))[:-futureDaysToPredicate]
        y = np.array(data['Prediction'])[:-futureDaysToPredicate]

        models = {
            'KNN': KNeighborsRegressor(n_neighbors=3),
            'K-MEANS': KMeans(n_clusters=2)
        }

        predicatedData = pd.DataFrame()

        for name, classifier in models.items():
            if name == 'K-MEANS':
                predicatedData[name] = classifier.fit(X)
            else:
                predicatedData[name] = calculatePredicateData(data, classifier, futureDaysToPredicate, X,y)

        return predicatedData


def loadStockSymbols():
    # get symbol list based on market
    url = "https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
    response = requests.get(url).content
    return pd.read_csv(io.StringIO(response.decode('utf-8')))['Symbol'].tolist()


def main():
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.title("Stock prediction app")

    daysForPredication = 7

    stockSymbols = loadStockSymbols()
    stockSymbols.insert(0, "...")
    st.selectbox("Select stock:",stockSymbols, index=0, key="item")

    if st.session_state.item != "...":
        v = getTimeRangeAsString(daysForPredication)
        print("date range to check if stock data exist:", getTimeRangeAsString(daysForPredication))
        dataForPredication = downloadStockDetails(st.session_state.item, v)
        if len(dataForPredication) == 0:
            st.text("The selected stock is delisted, please choose another stock")
        else:
            v = getTimeRangeAsString(getNumberOfDaysForTraining())
            print("date range for training:", v)
            data = downloadStockDetails(st.session_state.item, v)

            st.radio(
                "Select Classifier:",
                ["Decision Tree Classifier", "On Demand Classifier"],
                key="classifier",
                label_visibility='visible',
                horizontal=True
            )

            if st.session_state.classifier == "Decision Tree Classifier":
                predicatedData = precidcateBasedOnDecisionTreeClassifier(data)

            if st.session_state.classifier == "On Demand Classifier":
                predicatedData = precidcateBasedOnDemandClassifier(data)

            actualDataForPredication = pd.DataFrame(data['Close'][len(data) - daysForPredication:])
            actualDataForPredication.rename(columns={'Close': 'Original Price'}, inplace=True)
            print("columns:", actualDataForPredication.index)
            for column in predicatedData.columns:
                actualDataForPredication[column] = predicatedData[column][
                                                   actualDataForPredication.index[0]:actualDataForPredication.index[-1]]

            fig = px.line(actualDataForPredication, x=actualDataForPredication.index,
                          y=actualDataForPredication.columns)
            fig.update_layout(
                autosize=False,
                width=1000,
                height=500)
            fig.update_yaxes(title_text='Original Price')
            config = {'displayModeBar': False}
            # use_container_width=True,
            st.plotly_chart(fig, config=config)



if __name__ == '__main__':
    main()

