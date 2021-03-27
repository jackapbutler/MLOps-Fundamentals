import pandas as pd
data = pd.read_csv('./data/diabetes.csv')
data.to_csv('./data/processed_data.csv', index=False)