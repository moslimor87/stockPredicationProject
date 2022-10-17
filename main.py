from datetime import datetime
from dateutil import parser
from datetime import timedelta
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pytickersymbols import PyTickerSymbols
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}
</style>
"""

##get time range for training data
def getTimeRangeAsString(timeRange):
    # current date and time
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), (now - timedelta(days=timeRange)).strftime("%Y-%m-%d")

def getTimeRangeAsStringByGivenDate(date, timeRange):
    # current date and time
    return date, (parser.parse(date)- timedelta(days=timeRange)).strftime("%Y-%m-%d")

##get amount of data for dataset
def getNumberOfDaysForTraining():

    st.radio(
        "Select time range for dataset:",
        ["Week", "Month", "3 Months"],
        key="option",
        label_visibility= 'visible',
        horizontal= True
    )

    futureDaysToPredicate = 7

    if st.session_state.option == "Week":
        return futureDaysToPredicate + 7
    if st.session_state.option == "Month":
        return futureDaysToPredicate + 30
    if st.session_state.option == "3 Months":
        return futureDaysToPredicate + 90


##get historical stock data based on stock name and date range
def downloadStockDetails(stock, timeRangeForPredication):
    print('date range:', timeRangeForPredication)
    return yf.download(stock, timeRangeForPredication[1], timeRangeForPredication[0])


def calculatePredictedData(data, classifier, futureDaysToPredicate, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    classifier.fit(x_train, y_train)
    x_future = data.drop(['Prediction'], 1)[:-futureDaysToPredicate]
    x_future = x_future.tail(futureDaysToPredicate)
    x_future = np.array(x_future)

    tree_prediction = classifier.predict(x_future)
    predictions = tree_prediction
    valid = data[X.shape[0]:]
    valid['Predictions'] = predictions
    return valid['Predictions']


def precidcteStockDataBasedOnDecisionTreeClassifier(data):

    futureDaysToPredicate = 7
    data['Prediction'] = data['Close'].shift(-futureDaysToPredicate)
    X = np.array(data.drop(['Prediction'], 1))[:-futureDaysToPredicate]
    y = np.array(data['Prediction'])[:-futureDaysToPredicate]

    models = {
                'ID3': DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features="sqrt"),
                'CART': DecisionTreeClassifier(criterion="gini", max_depth=6, max_features="sqrt")
            }

    predicatedData = pd.DataFrame()

    for name, classifier in models.items():
        y = y.astype('int')
        predicatedData[name] = calculatePredictedData(data, classifier, futureDaysToPredicate, X,y)

    return predicatedData


def precidcteStockDataBasedOnDemandClassifier(data):

        futureDaysToPredicate = 7
        data['Prediction'] = data['Close'].shift(-futureDaysToPredicate)
        X = np.array(data.drop(['Prediction'], 1))[:-futureDaysToPredicate]
        y = np.array(data['Prediction'])[:-futureDaysToPredicate]

        models = {
            'KNN': KNeighborsRegressor(n_neighbors=3),
            'GNB': GaussianNB()
        }

        predicatedData = pd.DataFrame()

        for name, classifier in models.items():
            if name =='GNB':
                y = y.astype('int')
            predicatedData[name] = calculatePredictedData(data, classifier, futureDaysToPredicate, X,y)

        return predicatedData

## get selected classifier forn user and predict stock data
def predictStockDataBasedOnGivenClassifier(data):

    st.radio(
        "Select Classifier:",
        ["Decision Tree", "On Demand"],
        key="classifier",
        label_visibility='visible',
        horizontal=True
    )

    if st.session_state.classifier == "Decision Tree":
        return precidcteStockDataBasedOnDecisionTreeClassifier(data)

    if st.session_state.classifier == "On Demand":
        return precidcteStockDataBasedOnDemandClassifier(data)


## get stocks symbols form list
def loadStockSymbols():

    indexSymbols = {'FTSE 100': '^FTSE', 'S&P 500': '^GSPC', 'NASDAQ 100': 'NDX',
                     'DOW JONES': 'DJIA', 'S&P 100':'OEX', 'IBEX 35' :'IBEX', 'AEX': 'AEX',
                     'EURO STOXX 50': 'STOXX', 'Switzerland 20':'CH20'}
    # get symbol list based on market
    stock_data = PyTickerSymbols()

    stocks = []
    indexs = []
    indexBySectors = {}
    stockBySectors = {}
    for y in stock_data.get_stocks_by_country('United States'):
        if y["symbol"] not in stocks:
            stocks.append(y["symbol"])
        for industry in y['industries']:
            if stockBySectors.get(industry) is None:
                stockBySectors.update({industry: []})
            if y["symbol"] not in stockBySectors[industry]:
                stockBySectors[industry].append(y["symbol"])
            if indexBySectors.get(industry) is None:
                indexBySectors.update({industry: []})
            for x in y["indices"]:
                symbol = indexSymbols.get(x)
                if symbol not in indexs:
                    indexs.append(symbol)
                if symbol not in indexBySectors[industry]:
                    indexBySectors[industry].append(symbol)

    return stocks, indexs, stockBySectors, indexBySectors

def processRequest(symbols):

    daysForPredication = 7
    symbols.sort(reverse=False)
    symbols.insert(0, "...")
    st.selectbox("Select stock:", symbols, index=0, key="item")

    if st.session_state.item != "...":
        datRangeForCheckStockIsActive = getTimeRangeAsString(daysForPredication)
        dataForPredication = downloadStockDetails(st.session_state.item, datRangeForCheckStockIsActive)
        if len(dataForPredication) == 0:
            st.text("The selected stock is not active, please choose another stock")
        else:
            numberOfDays = getNumberOfDaysForTraining()
            dateRagneForTraining = getTimeRangeAsString(numberOfDays)
            dateRagneForFillTrainingData = getTimeRangeAsStringByGivenDate(dateRagneForTraining[1], 30)
            data = downloadStockDetails(st.session_state.item, dateRagneForTraining)
            missingDataInRange = downloadStockDetails(st.session_state.item, dateRagneForFillTrainingData)

            frames = [missingDataInRange[len(missingDataInRange) - (numberOfDays - len(data)):], data]
            result = pd.concat(frames)

            predicatedData = predictStockDataBasedOnGivenClassifier(result)

            actualDataForPredication = pd.DataFrame(data['Close'][len(data) - daysForPredication:])
            actualDataForPredication.rename(columns={'Close': 'Original'}, inplace=True)
            for column in predicatedData.columns:
                actualDataForPredication[column] = predicatedData[column][
                                                   actualDataForPredication.index[0]:actualDataForPredication.index[-1]]

            ##plot predict stock data results of classifier algorithms
            fig = px.line(actualDataForPredication, x=actualDataForPredication.index,
                          y=actualDataForPredication.columns)
            fig.update_layout(
                legend_title="",
                autosize=False,
                width=1000,
                height=500)
            fig.for_each_trace(
                lambda trace: trace.update(line=dict(color="Blue", width=8)) if trace.name == "Original" else (
                    trace.update(line=dict(color="Orange", width=10)) if trace.name == actualDataForPredication.columns[1] else (
                        trace.update(line=dict(color="Green", width=4)))))
            fig.update_yaxes(title_text='Close Price USD ($)')
            config = {'displayModeBar': False}
            st.plotly_chart(fig, config=config)


def setupMenuOptions(stockSymbols, indexSymbols, stocksByCategory, indexsByCetegory):
    st.radio(
        "Select stock by:",
        ["Symbol", "Index", "Category"],
        key="firstOption",
        label_visibility='visible',
        horizontal=True
    )

    if st.session_state.firstOption == "Category":
        categories = list(stocksByCategory.keys())
        categories.sort()
        categories.insert(0, "...")
        st.selectbox("Select category:", categories, index=0, key="category")

        if st.session_state.category != "...":
            st.radio(
                "Select by:",
                ["stock", "index"],
                key="categoryOption",
                label_visibility='visible',
                horizontal=True
            )

            if st.session_state.categoryOption == "index":
                processRequest(indexsByCetegory.get(st.session_state.category))
            if st.session_state.categoryOption == "stock":
                processRequest(stocksByCategory.get(st.session_state.category))

    if st.session_state.firstOption == "Symbol":
        processRequest(stockSymbols)

    if st.session_state.firstOption == "Index":
        processRequest(indexSymbols)


def main():
    st.set_page_config(layout="wide")
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.title("Predict Stock Market Trends")
    (stockSymbols, indicetsSymbols, stocksByCategory, indicesByCategory) = loadStockSymbols()
    setupMenuOptions(stockSymbols, indicetsSymbols, stocksByCategory, indicesByCategory)



if __name__ == '__main__':
    main()

