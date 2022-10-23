import pandas as pd
from config.configs import *
df_econ = pd.read_csv('/mnt/c/Users/Elijah/Downloads/CASH_FLOW.csv')
#df_fundamentals = pd.read_csv('/mnt/c/Users/Elijah/Downloads/FUNDAMENTAL_VALUATIONS.csv')

print(df_econ.head(5))
#print(df_fundamentals.head(5))

df_econ.to_sql('economy', con=POSTGRES_URL, schema='market', if_exists='append', index=False)
#df_fundamentals.to_sql('fundamental_valuations', con=POSTGRES_URL, schema='market', if_exists='append', index=False)