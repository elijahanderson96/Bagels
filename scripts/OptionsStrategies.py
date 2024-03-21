from typing import List

import pandas as pd


class OptionStrategy:
    def __init__(self, etf, account_balance, max_investment_percent):
        self.etf = etf
        self.account_balance = account_balance
        self.max_investment_percent = max_investment_percent

    def determine_option_parameters(self, row: pd.Series, trade_date: pd.Timestamp):
        raise NotImplementedError("Subclasses must implement this method.")

    def calculate_num_contracts_and_cost(
        self,
        current_stock_price: float,
        time_to_expiry: float,
        trade_date: pd.Timestamp,
    ):
        raise NotImplementedError("Subclasses must implement this method.")

    def execute_trade(
        self, row: pd.Series, num_contracts: int, premium_prices: List[float]
    ):
        raise NotImplementedError("Subclasses must implement this method.")


class SingleOptionStrategy(OptionStrategy):
    def determine_option_parameters(self, row: pd.Series, trade_date: pd.Timestamp):
        # Implementation for single option strategy
        pass

    def calculate_num_contracts_and_cost(
        self,
        current_stock_price: float,
        time_to_expiry: float,
        trade_date: pd.Timestamp,
    ):
        # Implementation for single option strategy
        pass

    def execute_trade(
        self, row: pd.Series, num_contracts: int, premium_prices: List[float]
    ):
        # Implementation for single option strategy
        pass


class IronCondorStrategy(OptionStrategy):
    def determine_option_parameters(self, row: pd.Series, trade_date: pd.Timestamp):
        # Implementation for iron condor strategy
        pass

    def calculate_num_contracts_and_cost(
        self,
        current_stock_price: float,
        time_to_expiry: float,
        trade_date: pd.Timestamp,
    ):
        # Implementation for iron condor strategy
        pass

    def execute_trade(
        self, row: pd.Series, num_contracts: int, premium_prices: List[float]
    ):
        # Implementation for iron condor strategy
        pass
