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
       "assetsToEquity",
       "assetTurnover",
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
       "filingDate" + INTERVAL '91 days' as date,
        "filingDate"  as date_prev,
"goodwillTotal",
"incomeNetPerWabsoSplitAdjusted",
"incomeNetPerWabsoSplitAdjustedYoyDeltaPercent",
"incomeNetPerWadsoSplitAdjusted",
"incomeNetPerWadsoSplitAdjustedYoyDeltaPercent",
"incomeNetPreTax",
"interestBurden",
"inventoryTurnover",
"investedCapital",
"investedCapitalGrowth" ,
"investedCapitalTurnover",
leverage,
"netDebt",
"netIncomeGrowth",
"netIncomeToRevenue",
"netWorkingCapital",
"netWorkingCapitalGrowth",
"nibclRevenueDeferredTurnover",
nopat,
"nopatGrowth",
"operatingCashFlowGrowth",
"operatingCashFlowInterestCoverage",
"operatingCfToRevenue" ,
"operatingReturnOnAssets",
"preferredEquityToCapital"   ,
"pretaxIncomeMargin"   ,
"priceAccountingPeriodEnd"  ,
"priceToRevenue"  ,
"pToBv",
"pToE",
"quickRatio" ,
"returnOnAssets" ,
"returnOnEquity"  ,
"revenueGrowth" ,
roce  ,
roic  ,
symbol,
"taxBurden",
"workingCapitalTurnover"
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
SELECT date, value as real_gdp
FROM market.economic
WHERE key='A191RL1Q225SBEA'
AND date > '01-01-2001'

-- @~fetch_fed_funds
SELECT date, value as fed_funds_rate
FROM market.economic
WHERE key='FEDFUNDS'
AND date > '01-01-2001'

-- @~fetch_comm_paper_outstanding
SELECT date, value as comm_paper_outstanding
FROM market.economic
WHERE key='COMPOUT'
AND date > '01-01-2001'

-- @~fetch_unemployment_claims
SELECT date, value as unemployment_claims
FROM market.economic
WHERE key='IC4WSA'
AND date > '01-01-2001'

-- @~fetch_cpi
SELECT date, value as cpi
FROM market.economic
WHERE key='CPIAUCSL'
AND date > '01-01-2001'

-- @~fetch_vehicle_sales
SELECT date, value as vehicle_sales
FROM market.economic
WHERE key='TOTALSA'
AND date > '01-01-2001'

-- @~fetch_unemployment_rate
SELECT date, value as unemployment_rate
FROM market.economic
WHERE key='UNRATE'
AND date > '01-01-2001'

-- @~fetch_industrial_production
SELECT date, value as industrial_productions
FROM market.economic
WHERE key='INDPRO'
AND date > '01-01-2001'

-- @~fetch_housing_starts
SELECT date, value as housing_starts
FROM market.economic
WHERE key='HOUST'
AND date > '01-01-2001'

-- @~fetch_num_total_employees
SELECT date, value as total_workers
FROM market.economic
WHERE key='PAYEMS'
AND date > '01-01-2001'

-- @~fetch_recession_probability
SELECT date, value as recession_probability
FROM market.economic
WHERE key='RECPROUSM156N'
AND date > '01-01-2001'

-- @~fetch_15Ymortgage_rates
SELECT date, value as fifteen_year_mortage_rate
FROM market.mortgage
WHERE key='MORTGAGE15US'
AND date > '01-01-2001'

-- @~fetch_5Ymortgage_rates
SELECT date, value as five_year_mortage_rate
FROM market.mortgage
WHERE key='MORTGAGE5US'
AND date > '01-01-2001'

-- @~fetch_30Ymortgage_rates
SELECT date, value as thirty_year_mortage_rate
FROM market.mortgage
WHERE key='MORTGAGE30US'
AND date > '01-01-2001'


