-- @~labels
SELECT date, close * shares_outstanding as marketcap, symbol
FROM market.stock_prices
WHERE symbol in SYMBOLS;

-- @~cash_flow
SELECT * FROM market.cash_flow
WHERE symbol in SYMBOLS
ORDER BY reportdate ASC;

-- @~shares_outstanding
SELECT symbol, shares_outstanding as so
FROM market.stock_prices
WHERE symbol in SYMBOLS;

-- @~time_series
SELECT date, close
FROM market.stock_prices
WHERE symbol='SYMBOL'
ORDER BY date ASC