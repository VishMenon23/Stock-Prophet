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
<img width="1440" alt="1_git" src="https://github.com/VishMenon23/Stock-Prophet/assets/122495446/649084bb-aa35-4d17-bd97-fe4323d77a6a">
<img width="1440" alt="2_git" src="https://github.com/VishMenon23/Stock-Prophet/assets/122495446/52348290-dfa1-466d-b8b7-0918b2485ad0">
<img width="1440" alt="3_git" src="https://github.com/VishMenon23/Stock-Prophet/assets/122495446/ab520182-7e9f-46c3-a7d2-ce3462eb5f76">
<img width="1440" alt="4_git" src="https://github.com/VishMenon23/Stock-Prophet/assets/122495446/ad89d88a-80f2-44ba-bcc0-bed0df20f95a">
