-- @~labels
SELECT date, close * shares_outstanding as marketcap, symbol
FROM market.stock_prices
WHERE symbol = SYMBOLS;

-- @~fundamentals
SELECT * FROM market.fundamentals
WHERE symbol = SYMBOLS
ORDER BY reportdate ASC;

-- @~shares_outstanding
SELECT symbol, shares_outstanding as so
FROM market.stock_prices
WHERE symbol =SYMBOLS;

-- @~time_series
SELECT date, close
FROM market.stock_prices
WHERE symbol='SYMBOL'
ORDER BY date ASC