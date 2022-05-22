def insert_prices(stock, token):
    """Grabs prices (our label data, and some features too) for a given stock. For now we'll use yfinance
    as it is quite convienent, and most importantly, free. This function accepts one argument, however,
    two are needed to pull the data: stock and beginning_date. Yfinance will pull all data up until the current date.
    The yfinance download method will then return a dataframe of the form: Date, Open, High, Low, Close, Adj close.
    We then tack on the symbol to the dataframe, since the results of the call will be inserted to our mysqldb and need to 
    Be categorized accordingly.

    Arguments: stock {str} -- Required. This is the stock you're fetching prices for.
    """
    import pandas as pd
    import pyEX
    from yfinance import download

    engine = 'postgresql://Elijah:Poodle13@localhost/auth'

    #stock_prices_df = pd.read_sql(f'SELECT Date FROM Stock_Prices WHERE symbol="{stock}";', engine).astype('datetime64')
    raw_financials_df = pd.read_sql(f'SELECT reportDate FROM fundamentals WHERE symbol="{stock}";', engine).astype(
        'datetime64')
    # Determine how far back we have to go, I.E. How much historical data do we need to bulk insert.
    beginning_date = min(raw_financials_df['reportDate'])
    #stock_prices_df = pd.DataFrame()
    #if not stock_prices_df.empty:
    #    last_date_in_stock_prices_table = max(stock_prices_df['Date'])
    #else:
    #    last_date_in_stock_prices_table = None

    # Our price history will go WAY to far back if beginning_date is None (I.E Raw Financials hasn't been populated
    # yet.) To circumvent this, we only grab prices if beginning_date is defined as a date, and not None. The dag
    # will need to be Re Ran otherwise.

    #if beginning_date is not None:
    #    if last_date_in_stock_prices_table is None:
    prices = download(stock, start=beginning_date)
        #else:
        #    prices = download(stock, start=last_date_in_stock_prices_table)

    shares_outstanding = pyEX.advancedStats(stock, token=token, filter='sharesOutstanding')['sharesOutstanding']

    prices['symbol'] = stock
    prices['sharesOutstanding'] = shares_outstanding
    prices['marketCap'] = prices['Close'] * shares_outstanding
    prices.to_sql('Stock_Prices', engine, if_exists='append')