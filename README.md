 # Stock Prophet: A Machine Learning Approach to Stock Price Prediction

Stock Prophet is a sophisticated web application designed to offer deep insights and predictions on stock prices using advanced machine learning techniques, specifically LSTM (Long Short-Term Memory) networks. This project leverages historical data from leading tech companies to forecast stock prices, assisting users in crafting informed investment strategies.

## Project Overview

Stock Prophet utilizes LSTM networks, a type of recurrent neural network, to identify long-term dependencies within time-series data effectively. The web application is developed using Django and provides a dynamic and interactive environment for users to explore and visualize potential future stock price movements.

## Key Features

- **Price Oracle**: Utilizes LSTM models to predict future stock prices based on historical data.
- **Graphical Dashboard**: Provides interactive graphs displaying current trends and historical performance of stock prices using the Plotly library.
- **Data Insights**: Offers a correlation matrix and analysis, helping users understand the interdependencies within the tech sector.


## Usage

After launching the server, access the web application via `localhost:8000` in your web browser. Explore the two main services provided:
- **Price Oracle**: Enter the stock ticker to get future price predictions.
- **Graphical Dashboard**: View interactive graphs showing stock trends and performance metrics.

## Data Sources

Data is primarily sourced from Yahoo Finance, covering top tech companies like Google, Apple, Microsoft, Nvidia, and Amazon, as well as major indices such as the S&P 500 and NASDAQ.

## Architecture

The backend of Stock Prophet is built on Django with LSTM models developed using TensorFlow's Keras API. The application also uses the Plotly library for rendering interactive charts and graphs.

## Screenshots of Website - 
