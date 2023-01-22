import logging
from datetime import date

import pandas as pd

from config.common import POSTGRES_URL
from config.mappings import ENERGY_SECTOR, FINANCE_SECTOR, HEALTHCARE_SECTOR
from data.iex import Pipeline
from data.labels import Prices
from models.models import PredictionPipeline

today = date.today()
# Month abbreviation, day and year
today = today.strftime("%b_%d_%Y")
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)

if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.update_data()
    prices = Prices()
    prices.update_db()
    for industry, stocks in {'financial': FINANCE_SECTOR,
                             'healthcare': HEALTHCARE_SECTOR,
                             'energy': ENERGY_SECTOR}.items():
        model = PredictionPipeline(symbols=stocks,
                                   model_type='classify',
                                   validate=False,
                                   features=['fundamental_valuations',
                                             'fetch_5Ymortgage_rates',
                                             'fetch_15Ymortgage_rates',
                                             'fetch_30Ymortgage_rates',
                                             'fetch_recession_probability',
                                             'fetch_num_total_employees',
                                             'fetch_housing_starts',
                                             'fetch_industrial_production',
                                             'fetch_unemployment_rate',
                                             'fetch_vehicle_sales',
                                             'fetch_cpi',
                                             'fetch_unemployment_claims',
                                             'fetch_comm_paper_outstanding',
                                             'fetch_fed_funds',
                                             'fetch_real_gdp',
                                             'fetch_crude_oil_brent',
                                             'fetch_henry_hub_natural_gas',
                                             'fetch_jet_fuel',
                                             'fetch_regular_conventional_gas',
                                             'fetch_midgrade_conventional_gas',
                                             'fetch_diesel',
                                             'fetch_gas_russia',
                                             'fetch_heating_oil',
                                             'fetch_crude_oil_wti',
                                             'fetch_propane',
                                             ])
        model.train()
        prediction_scores = model.predict()
        print(prediction_scores)
        prediction_scores.to_sql(schema='market',
                                 name=f'{industry}_predictions',
                                 con=POSTGRES_URL,
                                 if_exists='append'
                                 )
