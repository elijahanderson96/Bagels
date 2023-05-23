# tables have a variety of columns.
# This dict stores the mapping of what columns are preserved from raw to transform.
# It also allows columns to be pivoted (i.e. values of a column be made into columns themselves)
# and upsampled (dates inserted values interpolated).
tables = {
    'cash_flow':
        {'columns':
             ['symbol', 'reportDate', 'cashChange', 'capitalExpenditures', 'changesInReceivables',
              'changesInInventories', 'depreciation', 'cashFlowFinancing', 'cashFlow', 'netBorrowings',
              'netIncome', 'totalInvestingCashFlows'],
         'upsample': False
         },

    'treasury':
        {'columns':
             ['key', 'date', 'value'],
         'upsample': True,
         'upsample_kwargs':
             {'pivot': True,
              'columns': 'key',  # key column is going to be pivoted
              'values': 'value'},  # value column to be extracted is value
         },

    'economic':
        {'columns':
             ['key', 'date', 'value'],
         'upsample': True,
         'upsample_kwargs':
             {'pivot': True,
              'columns': 'key',  # key column is going to be pivoted
              'values': 'value'},  # value column to be extracted is value
         },

    'historical_prices':
        {'columns':
             ['symbol', 'date', 'open', 'high', 'low', 'close'],
         'upsample': False
         },

    'energy':
        {'columns':
             ['key', 'date', 'value'],
         'upsample': True,
         'upsample_kwargs':
             {'pivot': True,
              'columns': 'key',  # key column is going to be pivoted
              'values': 'value'},  # value column to be extracted is value
         },
    'rates':
        {'columns':
             ['key', 'date', 'value'],
         'upsample': True,
         'upsample_kwargs':
             {'pivot': True,
              'columns': 'key',  # key column is going to be pivoted
              'values': 'value'},  # value column to be extracted is value
         },
}


