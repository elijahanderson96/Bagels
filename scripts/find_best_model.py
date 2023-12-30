import pandas as pd
from etf_predictor import DatasetBuilder, ETFPredictor
import argparse
import itertools

from scripts.ingestion_fred import load_config


def grid_search(train_data, predict_data, params):
    results_list = []  # A list to store individual results

    for epoch, batch_size, stride in itertools.product(*params.values()):
        predictor = ETFPredictor(
            train_data=train_data,
            predict_data=predict_data,
            sequence_length=28,  # Adjust as needed
            epochs=epoch,
            batch_size=batch_size,
            stride=stride
        )
        mae = predictor.backtest(window_length=3000, overlap=2500, days_ahead=182)  # Adjust as needed
        results_list.append({'Epochs': epoch, 'Batch_Size': batch_size, 'Stride': stride, 'MAE': mae})

    results = pd.DataFrame(results_list)  # Convert the list of dictionaries to a DataFrame
    return results


def main():
    parser = argparse.ArgumentParser(description="Search for best model configurations.")
    parser.add_argument("--etf", type=str, required=True, help="The ETF we are sourcing data for.")

    args = parser.parse_args()

    # Load data
    file_path = f"./etf_feature_mappings/{args.etf.lower()}.yml"
    config = load_config(filename=file_path)
    endpoints = config["endpoints"]

    tables = [endpoint.lower() for endpoint in endpoints.keys()]
    dataset_builder = DatasetBuilder(table_names=tables, etf_symbol=args.etf, forecast_n_days_ahead=182)
    train_data, predict_data = dataset_builder.build_datasets()
    train_data.drop(columns="date_label", inplace=True)

    # Define the range of parameters for grid search
    params = {
        'Epochs': [10, 50, 100],
        'Batch_Size': [8, 16, 32],
        'Stride': [7, 14, 21]
    }

    # Run grid search
    search_results = grid_search(train_data, predict_data, params)

    # Identify the best configuration
    best_idx = search_results['MAE'].idxmin()
    search_results['Best_Config'] = ['Yes' if idx == best_idx else 'No' for idx in search_results.index]

    # Save results to CSV
    search_results.to_csv('model_search_results.csv', index=False)

    print("Grid search completed. Results saved to model_search_results.csv.")
    print(
        f"Best configuration (Epochs: {search_results.loc[best_idx, 'Epochs']}, Batch Size: {search_results.loc[best_idx, 'Batch_Size']}, Stride: {search_results.loc[best_idx, 'Stride']}) with MAE: {search_results.loc[best_idx, 'MAE']}"
        )


if __name__ == "__main__":
    main()
