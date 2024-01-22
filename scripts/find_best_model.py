import argparse
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor


def spawn_training_process(params, etf):
    # Convert parameters to command line arguments
    command = [
        "python",
        "scripts/etf_predictor.py",
        "--etf",
        etf,
        "--days_ahead",
        str(params["Days_Ahead"]),
        "--sequence_length",
        str(params["Sequence_Length"]),
        "--epochs",
        str(params["Epochs"]),  # Fixed number of epochs
        "--batch_size",
        str(params["Batch_Size"]),  # Fixed batch size
        "--stride",
        str(1),  # Stride is half of the sequence length
        "--window_length",
        str(params["Window_Length"]),
        "--overlap",
        str(params["Window_Length"] - 7),  # No overlap
        "--train",
        "--from_date",
        "2021-01-01",
        "--learning_rate",
        ".00005"
    ]
    subprocess.run(command)  # Using run for synchronous execution


def main():
    parser = argparse.ArgumentParser(description="Find Best Model Configuration")
    parser.add_argument(
        "--etf", type=str, required=True, help="ETF symbol for the model"
    )
    args = parser.parse_args()
    etf = args.etf

    # Define the range of parameters for grid search
    params = {
        "Days_Ahead": [112, 140, 210],
        "Sequence_Length": [14, 21, 28],
        "Window_Length": [150, 200, 250, 500, 1000],
        "Batch_Size": [32],
        "Epochs": [5000],
        "Learning Rate": [.000025]
    }

    # Using ProcessPoolExecutor to manage concurrent processes
    with ProcessPoolExecutor(
        max_workers=2
    ) as executor:  # Adjust max_workers based on your system
        futures = []
        for param_set in itertools.product(*params.values()):
            param_dict = dict(zip(params.keys(), param_set))
            future = executor.submit(spawn_training_process, param_dict, etf)
            futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("Grid search completed.")


if __name__ == "__main__":
    main()
