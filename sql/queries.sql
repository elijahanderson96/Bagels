-- @~labels
SELECT date, close * shares_outstanding as marketcap, symbol
FROM market.stock_prices
WHERE symbol in SYMBOLS;

-- @~cash_flow
SELECT *
FROM market.cash_flow
WHERE symbol in SYMBOLS;

-- @~fundamental_valuations
SELECT entry_id,
       "accountsReceivableTurnover",
       "altmanZScore",
       "bookValuePerShare",
       "cashConversionCycle",
       "currentRatio",
       "daysInAccountsPayable",
       "daysInInventory",
       "daysInRevenueDeferred",
       "debtToEbitda",
       "debtToEquity",
       "dividendPerShare",
       "dividendYield",
       "earningsYield",
       "ebitdaGrowth",
       "ebitdaMargin",
       "ebitGrowth",
       "ebitToInterestExpense",
       "ebitToRevenue",
       "evToEbit",
       "evToEbitda",
       "evToFcf",
       "evToInvestedCapital",
       "evToNopat",
       "evToOcf",
       "evToSales",
       "fcfYield",
       "fiscalQuarter",
       "fixedAssetTurnover",
       "freeCashFlow",
       "freeCashFlowGrowth",
       "freeCashFlowToRevenue",
       "filingDate" + INTERVAL '91 days' as date, "filingDate" as date_prev, "goodwillTotal", "incomeNetPerWabsoSplitAdjustedYoyDeltaPercent", "incomeNetPerWadsoSplitAdjusted", "incomeNetPerWadsoSplitAdjustedYoyDeltaPercent", "incomeNetPreTax", "interestBurden", "inventoryTurnover", "investedCapital", "investedCapitalGrowth", "investedCapitalTurnover", leverage, "netDebt", "netIncomeGrowth", "netIncomeToRevenue", "netWorkingCapital", "netWorkingCapitalGrowth", "nibclRevenueDeferredTurnover", nopat, "nopatGrowth", "operatingCashFlowGrowth", "operatingCashFlowInterestCoverage", "operatingCfToRevenue", "operatingReturnOnAssets", "pretaxIncomeMargin", "priceAccountingPeriodEnd", "priceToRevenue", "pToBv", "pToE", "quickRatio", "returnOnAssets", "returnOnEquity", "revenueGrowth", roce, roic, symbol, "taxBurden", "workingCapitalTurnover"
FROM market.fundamental_valuations
WHERE symbol in SYMBOLS
  AND subkey='ttm';

-- @~shares_outstanding
SELECT symbol, shares_outstanding as so
FROM market.stock_prices
WHERE symbol in SYMBOLS;

-- @~time_series
SELECT date, close
FROM market.stock_prices
WHERE symbol='SYMBOL'
ORDER BY date ASC

-- @~fetch_latest_prices
SELECT sp.symbol, sp.date as date_current_close, sp.close as current_close
FROM market.stock_prices sp
         INNER JOIN (SELECT symbol, max(date) as max_date FROM market.stock_prices GROUP BY symbol) ag
                    ON sp.symbol = ag.symbol AND sp.date = ag.max_date;


-- @~fetch_real_gdp
SELECT date + INTERVAL '91 days' as date, value as real_gdp
FROM market.economic
WHERE key ='A191RL1Q225SBEA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_fed_funds
SELECT date + INTERVAL '91 days' as date, value as fed_funds_rate
FROM market.economic
WHERE key ='FEDFUNDS'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_comm_paper_outstanding
SELECT date + INTERVAL '91 days' as date, value as comm_paper_outstanding
FROM market.economic
WHERE key ='COMPOUT'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_unemployment_claims
SELECT date + INTERVAL '91 days' as date, value as unemployment_claims
FROM market.economic
WHERE key ='IC4WSA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_cpi
SELECT date + INTERVAL '91 days' as date, value as cpi
FROM market.economic
WHERE key ='CPIAUCSL'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_vehicle_sales
SELECT date + INTERVAL '91 days' as date, value as vehicle_sales
FROM market.economic
WHERE key ='TOTALSA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_unemployment_rate
SELECT date + INTERVAL '91 days' as date, value as unemployment_rate
FROM market.economic
WHERE key ='UNRATE'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_industrial_production
SELECT date + INTERVAL '91 days' as date, value as industrial_productions
FROM market.economic
WHERE key ='INDPRO'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_housing_starts
SELECT date + INTERVAL '91 days' as date, value as housing_starts
FROM market.economic
WHERE key ='HOUST'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_num_total_employees
SELECT date + INTERVAL '91 days' as date, value as total_workers
FROM market.economic
WHERE key ='PAYEMS'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_recession_probability
SELECT date + INTERVAL '91 days' as date, value as recession_probability
FROM market.economic
WHERE key ='RECPROUSM156N'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_15Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as fifteen_year_mortage_rate
FROM market.mortgage
WHERE key ='MORTGAGE15US'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_5Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as five_year_mortage_rate
FROM market.mortgage
WHERE key ='MORTGAGE5US'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_30Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as thirty_year_mortage_rate
FROM market.mortgage
WHERE key ='MORTGAGE30US'
  AND date
    > '01-01-2001'
ORDER BY date


