import argparse
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor


def spawn_training_process(params, etf, model_index, total_models):
    # Display the current model being trained
    print(f"Training model {model_index} of {total_models}: {params}")

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
        str(params["Epochs"]),
        "--batch_size",
        str(params["Batch_Size"]),
        "--stride",
        str(1),
        "--window_length",
        str(params["Window_Length"]),
        "--overlap",
        str(params["Window_Length"] - 7),
        "--learning_rate",
        str(params["Learning_Rate"]),
        "--kernel_regularizer_l1",
        str(params["L1_Kernel_Regularizer"]),
        "--kernel_regularizer_l2",
        str(params["L2_Kernel_Regularizer"]),
        "--from_date",
        "2020-01-01",
        "--train"
    ]
    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser(description="Find Best Model Configuration")
    parser.add_argument(
        "--etf", type=str, required=True, help="ETF symbol for the model"
    )
    args = parser.parse_args()
    etf = args.etf

    # Define the range of parameters for grid search
    params = {
        "Days_Ahead": [140],
        "Sequence_Length": [8],
        "Window_Length": [512],
        "Batch_Size": [64],
        "Epochs": [25],
        "Learning_Rate": [0.003],
        "L1_Kernel_Regularizer": [.08],
        "L2_Kernel_Regularizer": [.1],

    }

    # Calculate total number of models
    total_models = len(list(itertools.product(*params.values())))
    print(f"Total models to train: {total_models}")

    # Using ProcessPoolExecutor to manage concurrent processes
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for model_index, param_set in enumerate(
                itertools.product(*params.values()), start=1
        ):
            param_dict = dict(zip(params.keys(), param_set))
            future = executor.submit(
                spawn_training_process, param_dict, etf, model_index, total_models
            )
            futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("Grid search completed.")


if __name__ == "__main__":
    main()
