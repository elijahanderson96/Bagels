import json
from time import sleep
from typing import Dict

import pandas as pd
from requests_html import HTMLSession

from database import db_connector


class EdgarClient:
    """Class to handle requests to the SEC's EDGAR database."""

    def __init__(self, base_url: str = 'https://data.sec.gov'):
        """
        Initialize the EdgarClient object.

        Args:
            base_url (str): The base URL for the SEC's EDGAR database. Default is 'https://data.sec.gov'.
        """
        self.base_url = base_url
        self.headers = {'User-Agent': 'MyApp/Version1.0 (Your Name; Your Email)'}
        self.ticker_to_cik = self._load_cik_mapping()
        self.session = HTMLSession()

    @staticmethod
    def _load_cik_mapping() -> Dict[str, str]:
        """
        Load the mapping from ticker symbols to CIKs from a JSON file.

        Returns:
            ticker_to_cik (Dict[str, str]): Dictionary mapping ticker symbols to CIKs.
        """
        with open("./data/cik_to_ticker.json", "r") as file:
            data = json.load(file)
        return {item["ticker"]: str(item["cik_str"]).zfill(10) for item in data.values()}

    def _get(self, endpoint: str) -> Dict:
        """
        Send a GET request to the specified endpoint and return the JSON response.

        Args:
            endpoint (str): The endpoint for the GET request.

        Returns:
            response (Dict): The JSON response.
        """
        url = f'{self.base_url}/{endpoint}'
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_submissions(self, ticker: str) -> Dict:
        """
        Get the submissions history by ticker symbol.

        Args:
            ticker (str): The ticker symbol for the company.

        Returns:
            response (Dict): The submissions history.
        """
        cik = self.ticker_to_cik[ticker]
        endpoint = f"submissions/CIK{cik}.json"
        return self._get(endpoint)

    def get_company_concept(self, ticker: str, concept: str) -> Dict:
        """
        Get the XBRL disclosures from a single company and concept.

        Args:
            ticker (str): The ticker symbol for the company.
            concept (str): The concept (a taxonomy and tag).

        Returns:
            response (Dict): The XBRL disclosures.
        """
        cik = self.ticker_to_cik[ticker]
        endpoint = f"api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        return self._get(endpoint)

    def get_company_facts(self, ticker: str) -> Dict:
        """
        Get all the company concepts data for a company.

        Args:
            ticker (str): The ticker symbol for the company.

        Returns:
            response (Dict): The company concepts data.
        """
        cik = self.ticker_to_cik[ticker]
        endpoint = f"api/xbrl/companyfacts/CIK{cik}.json"
        return self._get(endpoint)

    def get_frames(self, concept: str, unit: str, period: str) -> Dict:
        """
        Get one fact for each reporting entity that is last filed that most closely fits the calendrical period
        requested.

        Args:
            concept (str): The concept (a taxonomy and tag).
            unit (str): The unit of measure.
            period (str): The calendrical period.

        Returns:
            response (Dict): The facts.
        """
        endpoint = f"api/xbrl/frames/us-gaap/{concept}/{unit}/{period}.json"
        return self._get(endpoint)


edgar = EdgarClient()
# apple_submissions = edgar.get_submissions('AAPL')

# apple_facts = edgar.get_company_facts('AAPL')
# apple_facts['facts'].keys()

# I believe this contains all the data the company has reported on their reports.
# For example Assets, AssetsCurrent, Amortization, etc.
# apple_facts['facts']['us-gaap'].keys()

# Its important to note that these columns won't be reported in every quarterly report.
# apple_facts['facts']['us-gaap']['InventoryRawMaterialsAndPurchasedPartsNetOfReserves']['units']['USD']

# We could maybe iterate over apple_facts['facts']['us-gaap'] and call get_company_concept(AAPL, iter)
# To get all data for apple.

# apple = edgar.get_company_concept('AAPL', 'Assets')

# apple.keys() -> 'cik', 'taxonomy', 'tag', 'label', 'description', 'entityName', 'units'
# apple['units']['USD'][2]  # This gives the assets reported on either a 10-K or 10-Q for a given period.

columns = []
for symbol in ['AMZN', 'AAPL', 'NVDA', 'TSLA', 'META', 'GOOGL']:
    print(f"Fetching data for {symbol}")
    dataframes = []
    symbol_facts = edgar.get_company_facts(symbol)
    for fact in symbol_facts['facts']['us-gaap']:
        data = edgar.get_company_concept(symbol, fact)
        if 'USD' in data['units'].keys() and len(data['units']['USD']) > 100:
            fiscal_period = [entry['fp'] for entry in data['units']['USD']]
            fiscal_year = [entry['fy'] for entry in data['units']['USD']]
            values = [entry['val'] for entry in data['units']['USD']]
            data = pd.DataFrame(data={'fiscal_period': fiscal_period, 'fiscal_year': fiscal_year, fact[-62:]: values})
            data.drop_duplicates(subset=['fiscal_period', 'fiscal_year'], keep='last', inplace=True)
            dataframes.append(data)
            sleep(.1)

    # Set 'fiscal year' and 'fiscal period' as indices for each dataframe
    df_list = [df.set_index(['fiscal_year', 'fiscal_period']) for df in dataframes]

    # Concatenate the list of dataframes
    df_final = pd.concat(df_list, axis=1)
    columns.append(list(df_final.columns))
    # Reset the index if you want 'fiscal year' and 'fiscal period' as normal columns
    df_final.reset_index(inplace=True)
    df_final.sort_values(by=['fiscal_year', 'fiscal_period'], inplace=True)
    imputed_df = df_final.fillna(df_final.median())
    print("Inserting dataframe.")
    kwargs = {'name': f'{symbol.lower()}', 'schema': 'reports', 'if_exists': 'replace'}
    db_connector.insert_dataframe(df_final, **kwargs)
    kwargs = {'name': f'{symbol.lower()}_imputed', 'schema': 'reports', 'if_exists': 'replace'}
    db_connector.insert_dataframe(imputed_df, **kwargs)

sets = []
for list in columns:
    sets.append(set(list))

common_features = sets[0].intersection(*sets[1:])
