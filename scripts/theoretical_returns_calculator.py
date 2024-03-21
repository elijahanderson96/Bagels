import logging
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as si

from api.helpers import decompress_dataframe
from database.database import db_connector


class OptionPricingModel:
    def __init__(self, symbol: str, date: str, dividend_yield: float = 0.02):
        """
        Initializes the OptionPricingModel with the symbol of the stock, a database connector,
        and a specific date for backtesting.

        Args:
            symbol (str): Symbol of the stock to fetch historical prices for.
            date (str): The date up to which historical data and risk-free rate should be considered.
        """
        self.symbol = symbol
        self.date = pd.to_datetime(date)
        self.historical_prices = self.fetch_historical_prices()
        self.risk_free_interest_rate = self.fetch_risk_free_interest_rate()
        self.dividend_yield = dividend_yield

    def fetch_historical_prices(self, sample_size=325) -> List[float]:
        """
        Fetches historical closing prices for the stock from the database,
        up to the specified date.

        Returns:
            List[float]: List of historical closing prices.
        """
        query = f"""SELECT date, close 
                    FROM {self.symbol}.{self.symbol}_historical_prices
                    WHERE date <= '{self.date.strftime('%Y-%m-%d')}'
                    ORDER BY date DESC
                    LIMIT {sample_size}"""
        historical_prices_df = db_connector.run_query(query)
        historical_prices_df = historical_prices_df.set_index("date")
        historical_prices_df.index = pd.to_datetime(historical_prices_df.index)
        return historical_prices_df["close"].to_list()

    def fetch_risk_free_interest_rate(self) -> float:
        """
        Fetches the latest risk-free interest rate from the database up to the specified date.

        Returns:
            float: The latest risk-free interest rate.
        """
        query = f"""SELECT value 
                    FROM {self.symbol}.dgs10 
                    WHERE date <= '{self.date.strftime('%Y-%m-%d')}'
                    ORDER BY date DESC
                    LIMIT 1"""
        risk_free_interest_rate = db_connector.run_query(
            query, return_df=False, fetch_one=True
        )
        return risk_free_interest_rate / 100  # Convert to a decimal

    def calculate_historical_volatility(self) -> float:
        """
        Calculates the historical volatility of the stock based on historical prices.

        Returns:
            float: The annualized historical volatility of the stock.
        """
        historical_prices_array = np.array(self.historical_prices)
        daily_returns = np.log(
            historical_prices_array[1:] / historical_prices_array[:-1]
        )
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualize the volatility
        return volatility

    def black_scholes_american(
        self, S: float, K: float, T: float, option_type: str = "call"
    ) -> float:
        """
        Calculates an approximate price of an American call or put option using the Black-Scholes model adjusted for dividends,
        using the class's risk-free interest rate and dividend yield.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration (in years).
            option_type (str): Type of the option ('call' or 'put'). Defaults to 'call'.

        Returns:
            float: Approximate price of the option.
        """
        r = self.risk_free_interest_rate
        q = self.dividend_yield
        sigma = self.calculate_historical_volatility()
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            option_price = S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(
                -r * T
            ) * si.norm.cdf(d2, 0.0, 1.0)
        else:
            option_price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(
                -q * T
            ) * si.norm.cdf(-d1, 0.0, 1.0)
        return option_price


class OptionsReturnCalculator:
    def __init__(
        self,
        etf: str,
        backtest_results_dataframe: pd.DataFrame,
        initial_investment: float,
        max_investment_percent: float = 0.05,
        option_position="ITM",
    ):
        """
        Initializes the OptionsReturnCalculator with ETF symbol, backtest results, initial investment,
        and maximum investment percentage per option trade.

        Args:
            etf (str): The ETF symbol to trade options on.
            backtest_results_dataframe (pd.DataFrame): DataFrame containing backtest results.
            initial_investment (float): The initial investment amount.
            max_investment_percent (float): Maximum percentage of the portfolio to be invested in a single option.
        """
        self.etf = etf
        self.backtest_results_dataframe = backtest_results_dataframe
        self.initial_investment = initial_investment
        self.max_investment_percent = max_investment_percent
        self.account_balance = initial_investment
        self.option_position = option_position

    def calculate_option_returns(self):
        """
        Calculates the total returns from trading options based on the backtest results and stores each result in a pandas DataFrame.

        Returns:
            float, pandas.DataFrame: The total returns from the options trading and a DataFrame with details of each trade.
        """
        # Initialize a list to store the results of each iteration
        results_list = []

        for index, row in self.backtest_results_dataframe.iterrows():
            try:
                trade_date, current_stock_price = self._prepare_trade(row)
                (
                    strike,
                    time_to_expiry,
                    option_type,
                    trade,
                ) = self._determine_option_parameters(row, trade_date)
                if not trade:
                    continue
                (
                    num_contracts,
                    cost_of_buying,
                    premium_price,
                ) = self._calculate_num_contracts_and_cost(
                    current_stock_price, strike, time_to_expiry, option_type, trade_date
                )
                net_return, final_price = self._execute_trade(
                    row, strike, option_type, num_contracts, premium_price
                )
                self.account_balance += net_return

                # Store the trade details in a dictionary and add it to the list
                trade_details = {
                    "Trade Date": trade_date,
                    "Stock Price": current_stock_price,
                    "Strike Price": strike,
                    "Time to Expiry": time_to_expiry,
                    "Option Type": option_type,
                    "Number of Contracts": num_contracts,
                    "Cost of Buying": cost_of_buying,
                    "Premium Price": premium_price,
                    "Net Return": net_return,
                    "Premium Price at Expiration": final_price,
                    "Account Balance After Trade": self.account_balance,
                }
                results_list.append(trade_details)

            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
                continue

        # Create a DataFrame from the list of trade details
        results_df = pd.DataFrame(results_list)

        total_returns = self.account_balance - self.initial_investment
        return total_returns, results_df

    def _prepare_trade(self, row: pd.Series) -> Tuple[pd.Timestamp, float]:
        """
        Prepares the trade by determining the trade date and fetching the current stock price.

        Args:
            row (pd.Series): A single row from the backtest results DataFrame.

        Returns:
            Tuple[pd.Timestamp, float]: The trade date and current stock price.
        """
        trade_date = pd.to_datetime(row["prediction_date"]) + pd.DateOffset(months=4)
        current_stock_price = self._fetch_stock_price(trade_date)
        return trade_date, current_stock_price

    def _fetch_stock_price(self, date: pd.Timestamp) -> float:
        """
        Fetches the stock price for a given date.

        Args:
            date (pd.Timestamp): The date for which to fetch the stock price.

        Returns:
            float: The stock price on the given date.
        """
        query = (
            f"SELECT close FROM {self.etf}.{self.etf}_historical_prices WHERE date <= '"
            f"{date.strftime('%Y-%m-%d')}' ORDER BY date DESC LIMIT 1"
        )
        stock_price = db_connector.run_query(query, return_df=False, fetch_one=True)
        if stock_price is None:
            raise ValueError(f"No stock price found for {date}")
        return stock_price

    def _determine_option_parameters(
        self, row: pd.Series, trade_date: pd.Timestamp
    ) -> Tuple[float, float, str, bool]:
        """
        Determines the option parameters including strike price, time to expiry, option type,
        and whether to proceed with buying an option, based on the predicted percentage change.

        Args:
            row (pd.Series): A single row from the backtest results DataFrame.
            trade_date (pd.Timestamp): The trade date for the option.

        Returns:
            Tuple[float, float, str, bool]: Adjusted strike price, time to expiry (in years), option type,
            and a boolean indicating whether to proceed with the option trade.
        """
        predicted_close_date = pd.to_datetime(row["predicted_close_date"])
        delta = predicted_close_date - trade_date
        time_to_expiry = delta.days / 365.0  # Convert delta to years

        current_stock_price = self._fetch_stock_price(trade_date)
        predicted_price = row["predicted_close_price"]
        percentage_change = (
            (predicted_price - current_stock_price) / current_stock_price
        ) * 100

        # Define the threshold for deciding on the option trade
        threshold = 10
        # Check if the predicted change exceeds the threshold
        if abs(percentage_change) < threshold:
            # If the change does not meet the threshold, do not proceed with the trade
            return 0, 0, "none", False

        option_type = "call" if percentage_change > 0 else "put"

        # Adjust the strike price based on the option type and position (ITM or OTM)
        if option_type == "call":
            if self.option_position == "ITM":
                strike = (
                    current_stock_price - (current_stock_price % 5) - 15
                )  # Deeper ITM for calls
            else:  # OTM
                strike = (
                    current_stock_price - (current_stock_price % 5) + 15
                )  # Deeper OTM for calls
        else:  # 'put'
            if self.option_position == "ITM":
                strike = (
                    current_stock_price - (current_stock_price % 5) + 15
                )  # Deeper ITM for puts
            else:  # OTM
                strike = (
                    current_stock_price - (current_stock_price % 5) - 15
                )  # Deeper OTM for puts

        # Return the strike price, time to expiry, option type, and a boolean to indicate proceeding with the trade
        return strike, time_to_expiry, option_type, abs(percentage_change) >= threshold

    def _calculate_num_contracts_and_cost(
        self,
        current_stock_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: str,
        trade_date: pd.Timestamp,
    ) -> Tuple[int, float]:
        """
        Calculates the number of contracts that can be bought and their total cost.

        Args:
            current_stock_price (float): The current stock price.
            strike (float): Strike price of the option.
            time_to_expiry (float): Time to expiry in years.
            option_type (str): Type of the option ('call' or 'put').
            trade_date (pd.Timestamp): The trade date for the option.

        Returns:
            Tuple[int, float]: Number of contracts that can be bought and total cost.
        """
        model = OptionPricingModel(
            symbol=self.etf, date=trade_date.strftime("%Y-%m-%d")
        )
        premium_price = model.black_scholes_american(
            S=current_stock_price, K=strike, T=time_to_expiry, option_type=option_type
        )

        max_investment_per_option = self.account_balance * self.max_investment_percent
        contract_price = premium_price * 100  # Each contract controls 100 shares

        num_contracts = int(max_investment_per_option // contract_price)
        total_cost = num_contracts * contract_price

        return num_contracts, total_cost, premium_price

    def _execute_trade(
        self,
        row: pd.Series,
        strike: float,
        option_type: str,
        num_contracts: int,
        premium_price: float,
    ) -> float:
        """
        Executes the trade by simulating the selling of the options at the predicted close date and calculates the
        net return.

        Args:
            row (pd.Series): A single row from the backtest results DataFrame.
            strike (float): Strike price of the option.
            option_type (str): Type of the option ('call' or 'put').
            num_contracts (int): Number of contracts bought.
            premium_price (float): The price originally paid for the contract.

        Returns:
            float: Net return from the trade.
        """
        predicted_close_date = pd.to_datetime(row["predicted_close_date"])
        final_model = OptionPricingModel(
            symbol=self.etf, date=predicted_close_date.strftime("%Y-%m-%d")
        )
        final_price = final_model.black_scholes_american(
            S=row["actual_close_price"], K=strike, T=0.001, option_type=option_type
        )

        net_return = (
            (final_price - premium_price) * num_contracts * 100
        )  # Adjust for number of shares per contract
        return net_return, final_price


symbol = "spy"

query = f"""
    SELECT data_blob FROM spy.backtest_results WHERE model_id = 280
    """

result = db_connector.run_query(query)["data_blob"].squeeze()
df = decompress_dataframe(result).tail(n=10)
print(df)


calculator = OptionsReturnCalculator(
    symbol,
    df,
    initial_investment=124000,
    max_investment_percent=0.2,
    option_position="OTM",
)
total_returns, results = calculator.calculate_option_returns()

print(f"Total Returns: {total_returns}")
print(results)

# test = OptionPricingModel(symbol='SPY', date='2024-03-08')
# price = test.black_scholes_american(S=514.48, K=500, T=.083, option_type='call' )
# print(price)
