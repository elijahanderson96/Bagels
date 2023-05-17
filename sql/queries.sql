-- @~labels
SELECT date, close * shares_outstanding as marketcap, symbol
FROM bagels.stock_prices
WHERE symbol in SYMBOLS;

-- @~cash_flow
SELECT *
FROM bagels.cash_flow
WHERE symbol in SYMBOLS;

-- @~fundamental_valuations
SELECT "altmanZScore", "bookValuePerShare"
    "ebitToRevenue",
       "freeCashFlow",
       "filingDate" AS date,
       "priceToRevenue",
       "pToBv",
    "pToE",
    "quickRatio"
    "revenueGrowth",
    symbol
FROM bagels.fundamental_valuations
WHERE symbol in SYMBOLS
  AND subkey='ttm'
;

-- @~shares_outstanding
SELECT symbol, shares_outstanding as so
FROM bagels.stock_prices
WHERE symbol in SYMBOLS;

-- @~time_series
SELECT date, close
FROM bagels.stock_prices
WHERE symbol='SYMBOL'
ORDER BY date ASC

-- @~fetch_latest_prices
SELECT sp.symbol, sp.date as date_current_close, sp.close as current_close
FROM bagels.stock_prices sp
         INNER JOIN (SELECT symbol, max(date) as max_date FROM bagels.stock_prices GROUP BY symbol) ag
                    ON sp.symbol = ag.symbol AND sp.date = ag.max_date;


-- @~fetch_real_gdp
SELECT date + INTERVAL '56 days' as date, value as real_gdp
FROM bagels.economic
WHERE key ='A191RL1Q225SBEA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_fed_funds
SELECT date + INTERVAL '56 days' as date, value as fed_funds_rate
FROM bagels.economic
WHERE key ='FEDFUNDS'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_comm_paper_outstanding
SELECT date + INTERVAL '56 days' as date, value as comm_paper_outstanding
FROM bagels.economic
WHERE key ='COMPOUT'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_unemployment_claims
SELECT date + INTERVAL '56 days' as date, value as unemployment_claims
FROM bagels.economic
WHERE key ='IC4WSA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_cpi
SELECT date + INTERVAL '56 days' as date, value as cpi
FROM bagels.economic
WHERE key ='CPIAUCSL'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_vehicle_sales
SELECT date + INTERVAL '56 days' as date, value as vehicle_sales
FROM bagels.economic
WHERE key ='TOTALSA'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_unemployment_rate
SELECT date + INTERVAL '56 days' as date, value as unemployment_rate
FROM bagels.economic
WHERE key ='UNRATE'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_industrial_production
SELECT date + INTERVAL '56 days' as date, value as industrial_productions
FROM bagels.economic
WHERE key ='INDPRO'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_housing_starts
SELECT date + INTERVAL '91 days' as date, value as housing_starts
FROM bagels.economic
WHERE key ='HOUST'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_num_total_employees
SELECT date + INTERVAL '91 days' as date, value as total_workers
FROM bagels.economic
WHERE key ='PAYEMS'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_recession_probability
SELECT date + INTERVAL '112 days' as date, value as recession_probability
FROM bagels.economic
WHERE key ='RECPROUSM156N'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_15Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as fifteen_year_mortage_rate
FROM bagels.mortgage
WHERE key ='MORTGAGE15US'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_5Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as five_year_mortage_rate
FROM bagels.mortgage
WHERE key ='MORTGAGE5US'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_30Ymortgage_rates
SELECT date + INTERVAL '91 days' as date, value as thirty_year_mortage_rate
FROM bagels.mortgage
WHERE key ='MORTGAGE30US'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_propane
SELECT date + INTERVAL '56 days' as date, value as propane_prices
FROM bagels.energy
WHERE key = 'DPROPANEMBTX'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_crude_oil_wti
SELECT date + INTERVAL '56 days' as date, value as crude_oil_wti_prices
FROM bagels.energy
WHERE key = 'DCOILWTICO'
  AND date
    > '01-01-2001'
ORDER BY date


-- @~fetch_heating_oil
SELECT date + INTERVAL '56 days' as date, value as heating_oil_prices
FROM bagels.energy
WHERE key = 'DHOILNYH'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_gas_russia
SELECT date + INTERVAL '56 days' as date, value as gas_russia_prices
FROM bagels.energy
WHERE key = 'GASPRMCOVW'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_diesel
SELECT date + INTERVAL '56 days' as date, value as diesel_prices
FROM bagels.energy
WHERE key = 'GASDESW'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_midgrade_conventional_gas
SELECT date + INTERVAL '56 days' as date, value as midgrade_conventional_gas_prices
FROM bagels.energy
WHERE key = 'GASMIDCOVW'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_regular_conventional_gas
SELECT date + INTERVAL '56 days' as date, value as regular_conventional_gas_prices
FROM bagels.energy
WHERE key = 'GASREGCOVW'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_jet_fuel
SELECT date + INTERVAL '56 days' as date, value as jetfuel_prices
FROM bagels.energy
WHERE key = 'DJFUELUSGULF'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_henry_hub_natural_gas
SELECT date + INTERVAL '56 days' as date, value as henry_hub_natural_gas_prices
FROM bagels.energy
WHERE key = 'DHHNGSP'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_crude_oil_brent
SELECT date + INTERVAL '56 days' as date, value as europe_oil_prices
FROM bagels.energy
WHERE key = 'DCOILBRENTEU'
  AND date
    > '01-01-2001'
ORDER BY date

--FX RATES START HERE

-- @~fetch_usdthb
SELECT date + INTERVAL '56 days' as date, rate as usdthb
FROM bagels."fx-daily"
WHERE key = 'USDTHB'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdaed
SELECT date + INTERVAL '56 days' as date, rate as usdaed
FROM bagels."fx-daily"
WHERE key = 'USDAED'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdcny
SELECT date + INTERVAL '56 days' as date, rate as usdcny
FROM bagels."fx-daily"
WHERE key = 'USDCNY'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_nzdusd
SELECT date + INTERVAL '56 days' as date, rate as nzdusd
FROM bagels."fx-daily"
WHERE key = 'NZDUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdnok
SELECT date + INTERVAL '56 days' as date, rate as usdnok
FROM bagels."fx-daily"
WHERE key = 'USDNOK'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdsek
SELECT date + INTERVAL '56 days' as date, rate as usdsek
FROM bagels."fx-daily"
WHERE key = 'USDSEK'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdils
SELECT date + INTERVAL '56 days' as date, rate as usdils
FROM bagels."fx-daily"
WHERE key = 'USDILS'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_ethusd
SELECT date + INTERVAL '56 days' as date, rate as ethusd
FROM bagels."fx-daily"
WHERE key = 'ETHUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdtwd
SELECT date + INTERVAL '56 days' as date, rate as usdtwd
FROM bagels."fx-daily"
WHERE key = 'USDTWD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_audusd
SELECT date + INTERVAL '56 days' as date, rate as audusd
FROM bagels."fx-daily"
WHERE key = 'AUDUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdtry
SELECT date + INTERVAL '56 days' as date, rate as usdtry
FROM bagels."fx-daily"
WHERE key = 'USDTRY'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdmyr
SELECT date + INTERVAL '56 days' as date, rate as usdmyr
FROM bagels."fx-daily"
WHERE key = 'USDMYR'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdinr
SELECT date + INTERVAL '56 days' as date, rate as usdinr
FROM bagels."fx-daily"
WHERE key = 'USDINR'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdhkd
SELECT date + INTERVAL '56 days' as date, rate as usdinr
FROM bagels."fx-daily"
WHERE key = 'USDHKD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdczk
SELECT date + INTERVAL '56 days' as date, rate as usdczk
FROM bagels."fx-daily"
WHERE key = 'USDCZK'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdidr
SELECT date + INTERVAL '56 days' as date, rate as usdidr
FROM bagels."fx-daily"
WHERE key = 'USDIDR'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_xrpusd
SELECT date + INTERVAL '56 days' as date, rate as xrpusd
FROM bagels."fx-daily"
WHERE key = 'XRPUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdbhd
SELECT date + INTERVAL '56 days' as date, rate as usdbhd
FROM bagels."fx-daily"
WHERE key = 'USDBHD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_gbpusd
SELECT date + INTERVAL '56 days' as date, rate as gbpusd
FROM bagels."fx-daily"
WHERE key = 'GBPUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdbgn
SELECT date + INTERVAL '56 days' as date, rate as usdbgn
FROM bagels."fx-daily"
WHERE key = 'USDBGN'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdkwd
SELECT date + INTERVAL '56 days' as date, rate as usdkwd
FROM bagels."fx-daily"
WHERE key = 'USDKWD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdkrw
SELECT date + INTERVAL '56 days' as date, rate as usdkrw
FROM bagels."fx-daily"
WHERE key = 'USDKRW'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usddkk
SELECT date + INTERVAL '56 days' as date, rate as usddkk
FROM bagels."fx-daily"
WHERE key = 'USDDKK'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdcad
SELECT date + INTERVAL '56 days' as date, rate as usdcad
FROM bagels."fx-daily"
WHERE key = 'USDCAD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdzar
SELECT date + INTERVAL '56 days' as date, rate as usdzar
FROM bagels."fx-daily"
WHERE key = 'USDZAR'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdcnh
SELECT date + INTERVAL '56 days' as date, rate as usdcnh
FROM bagels."fx-daily"
WHERE key = 'USDCNH'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdpln
SELECT date + INTERVAL '56 days' as date, rate as usdpln
FROM bagels."fx-daily"
WHERE key = 'USDPLN'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdsgd
SELECT date + INTERVAL '56 days' as date, rate as usdsgd
FROM bagels."fx-daily"
WHERE key = 'USDSGD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_bchusd
SELECT date + INTERVAL '56 days' as date, rate as bchusd
FROM bagels."fx-daily"
WHERE key = 'BCHUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_ltcusd
SELECT date + INTERVAL '56 days' as date, rate as ltcusd
FROM bagels."fx-daily"
WHERE key = 'LTCUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_btcusd
SELECT date + INTERVAL '56 days' as date, rate as btcusd
FROM bagels."fx-daily"
WHERE key = 'BTCUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdjpy
SELECT date + INTERVAL '56 days' as date, rate as usdjpy
FROM bagels."fx-daily"
WHERE key = 'USDJPY'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdmxn
SELECT date + INTERVAL '56 days' as date, rate as usdmxn
FROM bagels."fx-daily"
WHERE key = 'USDMXN'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdhuf
SELECT date + INTERVAL '56 days' as date, rate as usdhuf
FROM bagels."fx-daily"
WHERE key = 'USDHUF'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdchf
SELECT date + INTERVAL '56 days' as date, rate as usdchf
FROM bagels."fx-daily"
WHERE key = 'USDCHF'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdsar
SELECT date + INTERVAL '56 days' as date, rate as usdsar
FROM bagels."fx-daily"
WHERE key = 'USDSAR'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdron
SELECT date + INTERVAL '56 days' as date, rate as usdron
FROM bagels."fx-daily"
WHERE key = 'USDRON'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_eurusd
SELECT date + INTERVAL '56 days' as date, rate as eurusd
FROM bagels."fx-daily"
WHERE key = 'EURUSD'
  AND date
    > '01-01-2001'
ORDER BY date

-- @~fetch_usdrub
SELECT date + INTERVAL '56 days' as date, rate as usdrub
FROM bagels."fx-daily"
WHERE key = 'USDRUB'
  AND date
    > '01-01-2001'
ORDER BY date











