import pandas as pd

csv_file_path = 'Stock_Prophet/app/Data/my_tickers.csv'

df = pd.read_csv(csv_file_path)

symbol_list = []

for symbol in df['Symbol']:
    symbol = symbol.strip()
    if symbol:  # This will be False for any string that is empty or only contains whitespace
        symbol_list.append(symbol)
print(symbol_list)
