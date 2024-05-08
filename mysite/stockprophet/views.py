import pandas as pd
import numpy as np
import json
import yfinance as yf
import datetime as dt
from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
import random
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# Home Page
def home(request):
    return render(request, 'home.html')

# Graphical Dashboard
def graphical_dashboard(request):
    data = yf.download(tickers=['AAPL', 'AMZN', 'GOOG', 'NVDA', 'MSFT'],group_by = 'ticker',threads=True,period='1y',interval='1d')
    data.reset_index(level=0, inplace=True)
    
    close_price_trend = go.Figure()
    close_price_trend.add_trace(go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL"))
    close_price_trend.add_trace(go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN"))
    close_price_trend.add_trace(go.Scatter(x=data['Date'], y=data['GOOG']['Adj Close'], name="GOOG"))
    close_price_trend.add_trace(go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA"))
    close_price_trend.add_trace(go.Scatter(x=data['Date'], y=data['MSFT']['Adj Close'], name="MSFT"))

    close_price_trend.update_layout(paper_bgcolor="white", plot_bgcolor="white",font_color="black",xaxis=dict(showgrid=True,                 gridcolor='black', gridwidth=1),yaxis=dict(showgrid=True, gridcolor='black', gridwidth=1),modebar=dict(bgcolor='black'))
    
    # Close price Chart
    plot_close_price_trend = plot(close_price_trend, auto_open=False, output_type='div')

    # OHLC Charts
    # AAPL
    fig = go.Figure(data=go.Ohlc(x=data['Date'],open=data['AAPL']['Open'],high=data['AAPL']['High'],low=data['AAPL']['Low'],
                    close=data['AAPL']['Close']))
    plot_div_aapl = plot(fig, auto_open=False, output_type='div')                    
    # AAPL
    # AMZN
    fig = go.Figure(data=go.Ohlc(x=data['Date'],open=data['AMZN']['Open'],high=data['AMZN']['High'],low=data['AMZN']['Low'],
                    close=data['AMZN']['Close']))
    plot_div_amzn = plot(fig, auto_open=False, output_type='div')                    
    # AMZN
    # GOOG
    fig = go.Figure(data=go.Ohlc(x=data['Date'],open=data['GOOG']['Open'],high=data['GOOG']['High'],low=data['GOOG']['Low'],
                    close=data['GOOG']['Close']))
    plot_div_goog = plot(fig, auto_open=False, output_type='div')                    
    # GOOG
    # NVDA
    fig = go.Figure(data=go.Ohlc(x=data['Date'],open=data['NVDA']['Open'],high=data['NVDA']['High'],low=data['NVDA']['Low'],
                    close=data['NVDA']['Close']))
    plot_div_nvda = plot(fig, auto_open=False, output_type='div')                    
    # NVDA
    # MSFT
    fig = go.Figure(data=go.Ohlc(x=data['Date'],open=data['MSFT']['Open'],high=data['MSFT']['High'],low=data['MSFT']['Low'],
                    close=data['MSFT']['Close']))
    plot_div_msft = plot(fig, auto_open=False, output_type='div')                    
    # MSFT
    
    df_aapl = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df_aapl.insert(0, "Ticker", "AAPL")
    df_amzn = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df_amzn.insert(0, "Ticker", "AMZN")
    df_goog = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df_goog.insert(0, "Ticker", "GOOGL")
    df_nvda = yf.download(tickers = 'NVDA', period='1d', interval='1d')
    df_nvda.insert(0, "Ticker", "NVDA")
    df_msft = yf.download(tickers = 'MSFT', period='1d', interval='1d')
    df_msft.insert(0, "Ticker", "MSFT")
    
    df_overall = pd.concat([df_aapl, df_amzn, df_goog, df_nvda, df_msft], axis=0)
    df_overall.reset_index(level=0, inplace=True)
    df_overall.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    dict_df = {'Date': object}
    df_overall = df_overall.astype(dict_df)
    df_overall.drop('Date', axis=1, inplace=True)
    json_df = df_overall.reset_index().to_json(orient ='records')
    stocks_val = []
    stocks_val = json.loads(json_df)
    return render(request, 'graphical_dashboard.html', {'plot_close_price_trend': plot_close_price_trend,'recent_stocks':stocks_val,
                 'plot_div_aapl': plot_div_aapl,'plot_div_amzn': plot_div_amzn, 'plot_div_goog': plot_div_goog,
                 'plot_div_nvda': plot_div_nvda, 'plot_div_msft': plot_div_msft})

# Stock Prophet Screen 1
def search(request):
    return render(request, 'search.html', {})

# Ticker Info Screen
def ticker(request):
    file_name = 'stockprophet/Data/my_tickers.csv'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    # df_ticker = pd.read_csv('./Data/my_tickers.csv') 
    df_ticker = pd.read_csv(absolute_path) 
    json_ticker = df_ticker.reset_index().to_json(orient ='records')
    list_ticker = []
    list_ticker = json.loads(json_ticker)
    return render(request, 'ticker.html', {'ticker_list': list_ticker})

def generate_dataset(original_data, look_back=1):
        data_X, data_Y = [], []
        for i in range(len(original_data)-look_back-1):
            a = original_data[i:(i+look_back), 0]
            data_X.append(a)
            data_Y.append(original_data[i + look_back, 0])
        return np.array(data_X), np.array(data_Y)

def stock_price_predict(ip):
    tickers = ip[0]
    data = yf.download(tickers, start='2023-01-01', end='2024-04-15')
    data.dropna(inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]
    look_back = 60
    X_test, y_test = generate_dataset(test_data, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    file_name = 'stockprophet/Lstm_Model/lstm_model.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)
    y_predict = lstm_model.predict(X_test.reshape(-1, X_test.shape[1]))
    y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1))
    print('Predicted Stock Price using the LSTM - ', y_predict)
    return y_predict[-1]

# Stock Prophet Screen 2
def predict(request, ticker_value="GOOG", number_of_days="100"):
    ticker_value = ticker_value.upper()
    df = yf.download(tickers = ticker_value, period='1d', interval='1m')
    number_of_days = int(number_of_days)
    last_row = df.iloc[-1] 
    print("LAST ROW", df)
    table_Open = last_row['Open']
    table_Close = last_row['Close']
    table_High = last_row['High']
    table_Low = last_row['Low']
    table_Volume = last_row['Volume']
    print(df.columns)
    
    from plotly.subplots import make_subplots
    colors = {"background": "#f5f5f5","text": "#000000"}

    df = yf.download(ticker_value, start="2023-01-01", end="2024-04-01")
    df.reset_index(inplace=True)    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    
    fig.add_trace(go.Bar(x=df["Date"],y=df["Volume"],name='Volume',), secondary_y=False)
    fig.add_trace(go.Line(x=df["Date"],y=df["Close"],name='Close',), secondary_y=True)
    fig.update_layout(plot_bgcolor=colors['background'],paper_bgcolor=colors['background'],font_color=colors['text'])
    fig.update_xaxes(rangeslider_visible=False,
                    rangeselector=dict(buttons=list([dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all",stepmode="backward"),])))

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes()
    fig.update_layout(template="plotly_white")
    volume_close_plot = plot(fig, auto_open=False, output_type='div')

    y_pred_avg = stock_price_predict([[ticker_value]])

    pred_dict = {"Date": [], "Prediction": []}
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black", 
                            xaxis=dict(showgrid=True,gridcolor='black',gridwidth=1),
                            yaxis=dict(showgrid=True,gridcolor='black',gridwidth=1),
                            modebar=dict(bgcolor='black'))
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')
    file_name = 'stockprophet/Data/my_tickers.csv'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    ticker = pd.read_csv(absolute_path)
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Close', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    return render(request, "result.html", context={ 'volume_close_plot': volume_close_plot, 
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'table_Open':table_Open,
                                                    'table_Close':table_Close,
                                                    'table_High':table_High,
                                                    'table_Low':table_Low,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':table_Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    'AvgPrice':y_pred_avg[-1]
                                                    })
