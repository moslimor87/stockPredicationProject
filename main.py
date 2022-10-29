from datetime import datetime
from dateutil import parser
from datetime import timedelta
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pytickersymbols import PyTickerSymbols
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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

classifierTuningParameters = {
    'decisionTree': {},
    'knn': {}
}

decisionTreeParamsForTuning = {
        "max_depth": range(1, 10),
        "min_samples_split": range(2, 10),
        "min_samples_leaf": range(1, 5),
}


##get tuning parameters for KNN algorithm
def tuneKnnHyperParameters(classifier, X, y):

    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 5))
    p = [1, 2]

    param_dict = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    grid = GridSearchCV(classifier, param_dict, cv=2, verbose=1, n_jobs=1)
    # Fit the model
    grid.fit(x_train, y_train)
    classifierTuningParameters['knn'] = grid.best_params_


##get tuning parameters for decision tree algorithms
def tuneDecisionTreeHyperParameters(classifier, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    grid = GridSearchCV(classifier, param_grid=decisionTreeParamsForTuning, cv=2, verbose=1, n_jobs=1)
    grid.fit(x_train, y_train)
    classifierTuningParameters['decisionTree'] = grid.best_params_


##get time range for training data
def getTimeRangeAsString(timeRange):
    # current date and time
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), (now - timedelta(days=timeRange)).strftime("%Y-%m-%d")


def getTimeRangeAsStringByGivenDate(date, timeRange):
    # current date and time
    return date, (parser.parse(date) - timedelta(days=timeRange)).strftime("%Y-%m-%d")


##get amount of data for dataset
def getNumberOfDaysForTraining():
    st.radio(
        "Select time range for dataset:",
        ["Week", "Month", "3 Months"],
        key="option",
        label_visibility='visible',
        horizontal=True
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


def getXYParamsBasedOnData(data):
    futureDaysToPredicate = 7
    data['Prediction'] = data['Close'].shift(-futureDaysToPredicate)
    X = np.array(data.drop(['Prediction'], 1))[:-futureDaysToPredicate]
    y = np.array(data['Prediction'])[:-futureDaysToPredicate]

    return X, y

def predictStockDataBasedOnDecisionTreeClassifier(dataForTraining, dataForTuning):

    futureDaysToPredicate = 7

    models = {
        'ID3': DecisionTreeClassifier(criterion="entropy"),
        'CART': DecisionTreeClassifier(criterion="gini")
    }

    predicatedData = pd.DataFrame()
    for name, classifier in models.items():
        if not bool(classifierTuningParameters['decisionTree']):
            X, Y = getXYParamsBasedOnData(dataForTuning)
            tuneDecisionTreeHyperParameters(classifier, X, Y.astype('int'))
        if name == 'ID3':
            classifier = DecisionTreeClassifier(criterion="entropy", max_depth=classifierTuningParameters['decisionTree']['max_depth'],
                                                min_samples_split=classifierTuningParameters['decisionTree']['min_samples_split'],
                                                min_samples_leaf=classifierTuningParameters['decisionTree']['min_samples_leaf'])#,
                                               # max_features="log2")
        if name == 'CART':
            classifier = DecisionTreeClassifier(criterion="gini", max_depth=classifierTuningParameters['decisionTree']['max_depth'],
                                                min_samples_split=classifierTuningParameters['decisionTree']['min_samples_split'],
                                                min_samples_leaf=classifierTuningParameters['decisionTree']['min_samples_leaf'])#,
                                               # max_features="log2")

        x, y = getXYParamsBasedOnData(dataForTraining)
        predicatedData[name] = calculatePredictedData(dataForTraining, classifier, futureDaysToPredicate, x, y.astype('int'))

    return predicatedData


def predictStockDataBasedOnDemandClassifier(dataForTraining, dataForTuning):
    futureDaysToPredicate = 7

    models = {
        'KNN': KNeighborsRegressor(),
        'GNB': GaussianNB()
    }

    predicatedData = pd.DataFrame()
    x, y = getXYParamsBasedOnData(dataForTraining)

    for name, classifier in models.items():

        if name == 'KNN':
            if not bool(classifierTuningParameters['knn']):
                X, Y = getXYParamsBasedOnData(dataForTuning)
                tuneKnnHyperParameters(classifier, X, Y)

            classifier = KNeighborsRegressor(n_neighbors=classifierTuningParameters['knn']['n_neighbors'],
                                            leaf_size=classifierTuningParameters['knn']['leaf_size'],
                                            p=classifierTuningParameters['knn']['p'])

        if name == 'GNB':
            y = y.astype('int')

        predicatedData[name] = calculatePredictedData(dataForTraining, classifier, futureDaysToPredicate, x, y)

    return predicatedData


## get selected classifier from user and predict stock data
def predictStockDataBasedOnGivenClassifier(dataForTraining, dataForTuning):
    st.radio(
        "Select Classifier:",
        ["Decision Tree", "On Demand"],
        key="classifier",
        label_visibility='visible',
        horizontal=True
    )

    if st.session_state.classifier == "Decision Tree":
        return predictStockDataBasedOnDecisionTreeClassifier(dataForTraining, dataForTuning)

    if st.session_state.classifier == "On Demand":
        return predictStockDataBasedOnDemandClassifier(dataForTraining, dataForTuning)


##get stocks symbols form list
def loadStockSymbols():
    indexSymbols = {'FTSE 100': '^FTSE', 'S&P 500': '^GSPC', 'NASDAQ 100': 'NDX',
                    'DOW JONES': 'DJIA', 'IBEX 35': 'IBEX'}
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

                if symbol is not None:
                    if symbol not in indexs:
                        indexs.append(symbol)
                if symbol not in indexBySectors[industry]:
                    indexBySectors[industry].append(symbol)

    return stocks, indexs, stockBySectors, indexBySectors


def processRequest(symbols):
    daysToPredicte = 7

    symbols.sort(reverse=False)
    symbols.insert(0, "...")
    st.selectbox("Select stock:", symbols, index=0, key="item")

    if st.session_state.item != "...":
        dateRangeForTraining = getTimeRangeAsString(150)
        dataForTuning = downloadStockDetails(st.session_state.item, dateRangeForTraining)
        if len(dataForTuning) == 0:
            st.text("The selected stock is not active, please choose another stock")
        else:
            numberOfDaysTraining = getNumberOfDaysForTraining()
            dataForTraining = dataForTuning[len(dataForTuning) - numberOfDaysTraining:]
            predicatedData = predictStockDataBasedOnGivenClassifier(dataForTraining, dataForTuning)
            actualDataForPredication = pd.DataFrame(dataForTraining['Close'][len(dataForTraining) - daysToPredicte:])
            actualDataForPredication.rename(columns={'Close': 'Original'}, inplace=True)
            for column in predicatedData.columns:
                actualDataForPredication[column] = predicatedData[column][
                                               actualDataForPredication.index[0]:actualDataForPredication.index[-1]]

            ##plot predict stock data results of classifier algorithms
            fig = px.line(actualDataForPredication, x=actualDataForPredication.index, y=actualDataForPredication.columns)
            fig.update_layout(
                legend_title="",
                autosize=False,
                width=1000,
                height=500)
            fig.for_each_trace(
                lambda trace: trace.update(line=dict(color="Blue", width=12)) if trace.name == "Original" else (
                    trace.update(line=dict(color="Orange", width=10)) if trace.name == actualDataForPredication.columns[1]
                    else (trace.update(line=dict(color="Green", width=4)))))
            fig.update_yaxes(title_text='Close Price USD ($)')
            config = {'displayModeBar': False}
            st.plotly_chart(fig, config=config)

# select stock by symbol/index/category and process request by selection
def displayMenuOptions(stockSymbols, indexSymbols, stocksByCategory, indexByCategory):
    st.radio(
        "Select stock by:",
        ["Symbol", "Index", "Category"],
        key="firstOption",
        label_visibility='visible',
        horizontal=True
    )

    if st.session_state.firstOption == "Symbol":
        processRequest(stockSymbols)

    if st.session_state.firstOption == "Index":
        processRequest(indexSymbols)

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

            selectedOption = None
            if st.session_state.categoryOption == "index":
                selectedOption = indexByCategory.get(st.session_state.category)
            if st.session_state.categoryOption == "stock":
                selectedOption = stocksByCategory.get(st.session_state.category)

            if selectedOption is not None:
                processRequest(selectedOption)
            else:
                st.text("The selected option is not active, please choose another one")


def main():
    st.set_page_config(layout="wide")
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.title("Predict Stock Market Trends")
    (stockSymbols, indexSymbols, stocksByCategory, indexsByCategory) = loadStockSymbols()
    displayMenuOptions(stockSymbols, indexSymbols, stocksByCategory, indexsByCategory)


if __name__ == '__main__':
    main()
