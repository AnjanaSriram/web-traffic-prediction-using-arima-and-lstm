import streamlit as st
import pandas as pd
import datetime
import numpy as np
import re
import matplotlib.pyplot as plt
import math
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')

image = './undraw_fast_loading_0lbh.png'

st.title("Web Traffic Forecasting")
st.header("Problem Statement")
st.write("Web traffic congestion is a scenario faced by network appliction frequently.")
st.write(
    "Web traffic congestion is a phenomenum where the number of requests to be fulfilled by the server increases beyond the resource allocation.")
st.image("undraw_fast_loading_0lbh.png", caption=None, width=300, use_column_width=None)
st.header("Solution")
st.write(
    "A prediction model which analyzes the web traffic pattern of the target server and thus the server resources are allocated in advance to handle the server load.")

path_name = st.sidebar.text_input("Enter file path for report")

st.sidebar.header("Visualisation Settings")
uploaded_file = st.sidebar.file_uploader(label="Upload your web traffic dataset(.csv)", type=['csv', 'xslx'])

st.sidebar.image("undraw_Data_trends_re_2cdy.png", caption=None, width=300, use_column_width=None)

global df

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file).fillna(0)

        list_of_column_names = list(df.columns)
        list_of_column_names.remove('Page')

        j = 0

        for i in range(1, len(list_of_column_names)):
            if i % 100 == 0:
                list_of_column_names[j] = list_of_column_names[i]
                j = j + 1

        x = [datetime.date(2015, 7, 1)] * 550


        def find_language(url):
            res = re.search('[a-z][a-z].wikipedia.org', url)
            if res:
                return res[0][0:2]
            return 'na'


        df['lang'] = df.Page.map(find_language)

        lang_sets = {}

        lang_sets['en'] = df[df.lang == 'en'].iloc[:, 0:-1]
        lang_sets['ja'] = df[df.lang == 'ja'].iloc[:, 0:-1]
        lang_sets['de'] = df[df.lang == 'de'].iloc[:, 0:-1]
        lang_sets['na'] = df[df.lang == 'na'].iloc[:, 0:-1]
        lang_sets['fr'] = df[df.lang == 'fr'].iloc[:, 0:-1]
        lang_sets['zh'] = df[df.lang == 'zh'].iloc[:, 0:-1]
        lang_sets['ru'] = df[df.lang == 'ru'].iloc[:, 0:-1]
        lang_sets['es'] = df[df.lang == 'es'].iloc[:, 0:-1]

        sums = {}
        for key in lang_sets:
            sums[key] = lang_sets[key].iloc[:, 1:].sum(axis=0) / lang_sets[key].shape[0]

        days = [r for r in range(sums['en'].shape[0])]

        fig = plt.figure(1, figsize=[10, 10])
        fig, ax = plt.subplots()
        plt.ylabel('Views per Page')
        plt.xlabel('Day')
        plt.title('Pages in Different Languages')
        labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
                  'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
                  'ru': 'Russian', 'es': 'Spanish'
                  }

        for key in sums:
            plt.plot(days, sums[key], label=labels[key])

        for key in sums:
            fig = plt.figure(1, figsize=[10, 5])
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            data = np.array(sums[key])
            autocorr = acf(data)
            pac = pacf(data)

            x = [x for x in range(len(pac))]
            ax1.plot(x[1:], autocorr[1:])

            ax2.plot(x[1:], pac[1:])
            ax1.set_xlabel('Lag')
            ax1.set_ylabel('Autocorrelation')

            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Partial Autocorrelation')

        warnings.filterwarnings('ignore')

        path_name.replace("\"", "\\")
        if path_name:
            file_name = path_name + "\\hits.txt"
        else:
            file_name = "..\\hits.txt"

        f = open(file_name, "w")

        list_of_column_names = list(df.columns)
        list_of_column_names.remove('Page')

        params = {'en': [4, 1, 0], 'ja': [7, 1, 1], 'de': [7, 1, 1], 'na': [4, 1, 0], 'fr': [4, 1, 0], 'zh': [7, 1, 1],
                  'ru': [4, 1, 0], 'es': [7, 1, 1]}

        lang_obj = {'en': 'English', 'ja': 'Japanese', 'de': 'German', 'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
                    'ru': 'Russian', 'es': 'Spanish'}

        for key in sums:
            for keys in lang_obj:
                if keys == key:
                    f.write(lang_obj[keys])
                    f.write("\n")

            data = np.array(sums[key])
            f.write("Date         Expected         Predicted         Error")
            f.write("\n")

            result = None
            arima = ARIMA(data, params[key])
            result = arima.fit(disp=False)
            pred = result.predict(2, 599, typ='levels')

            x = [i for i in range(600)]
            i = 0
            for i in range(len(data)):
                f.write(str(list_of_column_names[i]) + "     " + str(math.ceil(data[i])) + "               " + str(
                    math.ceil(pred[i])) + "           " + str(
                    np.sqrt(np.mean((math.ceil(data[i]) - math.ceil(pred[i])) ** 2))))
                f.write("\n")

        warnings.filterwarnings('ignore')

        train_df = df.drop('Page', axis=1)

        global chart
        for key in sums:
            row = [0] * sums[key].shape[0]
            for i in range(sums[key].shape[0]):
                row[i] = sums[key][i]

            X = row[0:549]
            y = row[1:550]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            sc = MinMaxScaler()
            X_train = np.reshape(X_train, (-1, 1))
            y_train = np.reshape(y_train, (-1, 1))
            X_train = sc.fit_transform(X_train)
            y_train = sc.fit_transform(y_train)

            X_train = np.reshape(X_train, (384, 1, 1))

            regressor = Sequential()

            regressor.add(LSTM(units=8, activation='relu', input_shape=(None, 1)))

            regressor.add(Dense(units=1))

            regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

            regressor.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)

            inputs = X
            inputs = np.reshape(inputs, (-1, 1))
            inputs = sc.transform(inputs)
            inputs = np.reshape(inputs, (549, 1, 1))
            y_pred = regressor.predict(inputs)
            y_pred = sc.inverse_transform(y_pred)

            b = np.reshape(y_pred, (np.product(y_pred.shape),))

            d = {'Expected': y, 'Predicted': b}
            chart = pd.DataFrame(data=d)
            for keys in lang_obj:
                if keys == key:
                    st.write("Web Traffic Prediction for " + lang_obj[keys] + " pages")

            progress_bar = st.sidebar.progress(0)
            chart_new = st.line_chart(chart[:25])
            j = 12
            for i in range(0, len(chart), 25):
                progress_bar.progress(j + 4)
                end_index = i + 25
                new_rows = chart[i:end_index]
                chart_new.add_rows(new_rows)
                time.sleep(0.4)
                j = j + 4
            progress_bar.empty()

    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)
