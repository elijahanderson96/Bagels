import os
import json
import glob
import argparse

def find_best_configuration(results_dir, etf_name, target_param, target_value):
    best_config = None
    best_metric = float('inf')

    etf_results_dir = os.path.join(results_dir, etf_name)

    for file_name in glob.glob(os.path.join(etf_results_dir, '*.json')):
        with open(file_name, 'r') as file:
            data = json.load(file)

            if data.get(target_param, None) == target_value:
                metric = data['results']['mae'] if 'mae' in data['results'] else float('inf')
                config = data

                if metric < best_metric:
                    best_metric = metric
                    best_config = config

    return best_config, best_metric

def main():
    parser = argparse.ArgumentParser(description="Find the best configuration for a specific ETF")
    parser.add_argument("--etf_name", type=str, required=True, help="ETF name to search results for")
    parser.add_argument("--target_param", type=str, required=True, help="Target parameter to filter results")
    parser.add_argument("--target_value", type=int, required=True, help="Value of the target parameter")

    args = parser.parse_args()

    results_dir = './grid_search_results'
    etf_name = args.etf_name
    target_param = args.target_param
    target_value = args.target_value

    best_config, best_metric = find_best_configuration(results_dir, etf_name, target_param, target_value)

    if best_config is not None:
        print(f"Best Configuration for {etf_name} with {target_param}={target_value}:")
        print(f"Configuration: {best_config}")
        print(f"Best Metric (MAE): {best_metric}")
    else:
        print(f"No results found for {etf_name} with {target_param}={target_value}.")

if __name__ == "__main__":
    main()
