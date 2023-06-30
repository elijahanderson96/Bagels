import time
import requests
import json
from typing import Optional, Dict
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class EdgarClient:
    """Class to handle requests to the SEC's EDGAR database."""

    def __init__(self, base_url: str = 'https://data.sec.gov', headers: Optional[Dict[str, str]] = None):
        """
        Initialize the EdgarClient object.

        Args:
            base_url (str): The base URL for the SEC's EDGAR database. Default is 'https://data.sec.gov'.
            headers (Dict[str, str]): Optional dictionary of HTTP headers.
        """
        self.base_url = base_url
        self.headers = headers or {
            'User-Agent': 'Sample Company Name AdminContact@<sample company domain>.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov',
        }
        self.ticker_to_cik = self._load_cik_mapping()
        self.session = self._init_session()

    def _load_cik_mapping(self) -> Dict[str, str]:
        """
        Load the mapping from ticker symbols to CIKs from a JSON file.

        Returns:
            ticker_to_cik (Dict[str, str]): Dictionary mapping ticker symbols to CIKs.
        """
        with open("./data/cik_to_ticker.json", "r") as file:
            data = json.load(file)
        return {item["ticker"]: str(item["cik_str"]).zfill(10) for item in data.values()}

    def _init_session(self) -> requests.Session:
        """
        Initialize a requests Session object with retry parameters and headers.

        Returns:
            session (requests.Session): Initialized Session object.
        """
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.headers.update(self.headers)
        return session

    def _get(self, endpoint: str) -> Dict:
        """
        Send a GET request to the specified endpoint and return the JSON response.

        Args:
            endpoint (str): The endpoint for the GET request.

        Returns:
            response.json (Dict): The JSON response.
        """
        url = f'{self.base_url}/{endpoint}'
        print(url)
        input('break')
        response = self.session.get(url)
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
        Get one fact for each reporting entity that is last filed that most closely fits the calendrical period requested.

        Args:
            concept (str): The concept (a taxonomy and tag).
            unit (str): The unit of measure.
            period (str): The calendrical period.

        Returns:
            response (Dict): The facts.
        """
        endpoint = f"api/xbrl/frames/us-gaap/{concept}/{unit}/{period}.json"
        return self._get(endpoint)



# Initialize the EdgarClient object
edgar = EdgarClient()

# Get the submissions history for Apple
apple_submissions = edgar.get_submissions('AAPL')
print("Apple Submissions:")
print(apple_submissions)

# Get the XBRL disclosures from Apple for the concept 'AccountsPayableCurrent'
apple_disclosures = edgar.get_company_concept('AAPL', 'AccountsPayableCurrent')
print("\nApple Disclosures for 'AccountsPayableCurrent':")
print(apple_disclosures)

# Get all the company concepts data for Apple
apple_facts = edgar.get_company_facts('AAPL')
print("\nApple Facts:")
print(apple_facts)

# Get the fact for each reporting entity that last filed 'AccountsPayableCurrent' for the period 'CY2019Q1I'
# Note: 'USD' is the unit of measure in this case
fact = edgar.get_frames('AccountsPayableCurrent', 'USD', 'CY2019Q1I')
print("\nFact for 'AccountsPayableCurrent' for 'CY2019Q1I':")
print(fact)
