-- @~labels
SELECT date, close * shares_outstanding as marketcap, symbol
FROM market.stock_prices
WHERE symbol in SYMBOLS;

-- @~cash_flow
SELECT * FROM market.cash_flow
WHERE symbol in SYMBOLS;

-- @~fundamental_valuations
SELECT "accountsPayableTurnover" ,
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
"daysRevenueOutstanding",
"debtToAssets",
"debtToCapitalization",
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
"filingDate" as date,
"goodwillTotal",
"incomeNetPerWabso",
"incomeNetPerWabsoSplitAdjusted",
"incomeNetPerWabsoSplitAdjustedYoyDeltaPercent",
"incomeNetPerWadso",
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
"netDebtToEbitda",
"netIncomeGrowth",
"netIncomeToRevenue",
"netWorkingCapital",
"netWorkingCapitalGrowth",
"nibclRevenueDeferredTurnover",
nopat,
"nopatGrowth",
"nopatMargin",
"operatingCashFlowGrowth",
"operatingCashFlowInterestCoverage",
"operatingCfToRevenue" ,
"operatingIncomeToRevenue",
"operatingReturnOnAssets",
"preferredEquityToCapital"   ,
"pretaxIncomeMargin"   ,
"priceAccountingPeriodEnd"  ,
"priceToRevenue"  ,
"profitGrossToRevenue",
"pToBv",
"pToE",
"quickRatio" ,
"researchDevelopmentToRevenue",
"returnOnAssets" ,
"returnOnEquity"  ,
"revenueGrowth" ,
roce  ,
roic  ,
"sgaToRevenue",
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