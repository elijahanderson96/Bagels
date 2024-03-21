from data.fred_data import DatasetBuilder
from database.database import db_connector
from forecasting.etf_predictor import ETFPredictor
from scripts.ingestion_fred import load_config

if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Build model for a given ETF with specified arguments"
    )

    # Add arguments
    parser.add_argument("--etf", type=str, help="The ETF we are sourcing data for.")
    parser.add_argument(
        "--days_ahead",
        type=int,
        default=182,
        help="How many days ahead are we forecasting?",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Are we training a model for current predictions?",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=28,
        help="Input sequence length for the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for model training."
    )
    parser.add_argument(
        "--stride", type=int, default=14, help="Stride for training data preparation."
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=None,
        help="Length of the training window for backtesting.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for model training",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2500,
        help="Number of overlapping days in the training window for backtesting.",
    )
    parser.add_argument(
        "--kernel_regularizer_l1",
        type=float,
        default=0.01,
        help="The value passed to the kernel_regularizer for L1 (Lasso) Regularization",
    )
    parser.add_argument(
        "--kernel_regularizer_l2",
        type=float,
        default=0.01,
        help="The value passed to the kernel_regularizer for L2 (Ridge) Regularization",
    )
    parser.add_argument("--from_date", type=str, default="2000-01-01")
    parser.add_argument(
        "--validate", action="store_true", help="Enable validation during training."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extracting arguments
    etf_arg = args.etf
    days_ahead = args.days_ahead
    train = args.train
    sequence_length = args.sequence_length
    epochs = args.epochs
    batch_size = args.batch_size
    stride = args.stride
    window_length = args.window_length
    overlap = args.overlap

    file_path = f"./etf_feature_mappings/{etf_arg.lower()}.yml"
    config = load_config(filename=file_path)
    endpoints = config["endpoints"]

    tables = [endpoint.lower() for endpoint in endpoints.keys()]

    self = DatasetBuilder(
        table_names=tables,
        etf_symbol=etf_arg,
        forecast_n_days_ahead=days_ahead,
        sequence_length=sequence_length,
        from_date=args.from_date,
    )

    train_data, predict_data = self.build_datasets()

    # # Backtest is performed by default
    # backtest_predictor = ETFPredictor(
    #     train_data=train_data,
    #     predict_data=predict_data,
    #     sequence_length=args.sequence_length,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     stride=args.stride,
    #     learning_rate=args.learning_rate,
    #     overlap=args.overlap,
    #     window_length=args.window_length,
    #     l1_kernel_regularizer=args.kernel_regularizer_l1,
    #     l2_kernel_regularizer=args.kernel_regularizer_l2
    #
    # )
    #
    # # Perform the backtest
    # analyzed_results, calculated_pmae, calculated_pmae_high, calculated_pmae_low = backtest_predictor.backtest(
    #     window_length=args.window_length,
    #     overlap=args.overlap,
    #     days_ahead=args.days_ahead,
    #
    # )
    # print(analyzed_results)
    # print(f"The mean absolute error percentage is: {calculated_pmae}%")
    # print(f"The mean absolute error percentage for the high: {calculated_pmae_high}%")
    # print(f"The mean absolute error percentage for the low: {calculated_pmae_low}%")

    predictor = ETFPredictor(
        etf=etf_arg,
        features=endpoints,
        train_data=train_data,
        predict_data=predict_data,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        window_length=args.window_length,
        learning_rate=args.learning_rate,
        overlap=args.overlap,
        from_date=args.from_date,
        days_ahead=args.days_ahead,
        l1_kernel_regularizer=args.kernel_regularizer_l1,
        l2_kernel_regularizer=args.kernel_regularizer_l2,
    )

    predictor.train(backtest=True)
    predictor.predict()
    predictor.save_experiment()

    # # Assuming prediction_df contains a single entry with a date and price
    # predicted_price = prediction_df["Predicted_Close"].iloc[0]
    # predicted_high = prediction_df["Predicted_High"].iloc[0]
    # predicted_low = prediction_df["Predicted_Low"].iloc[0]
    #
    # prediction_range = predictor.adjusted_prediction_range(
    #     predicted_price, predicted_high, predicted_low, calculated_pmae, calculated_pmae_high, calculated_pmae_low
    # )
    # print(prediction_range)
    # print(
    #     f"Prediction Range Based on MAE alone: {prediction_range[0]} to {prediction_range[1]}"
    # )
    # print(
    #     f"Predicted Date: {prediction_df['Date'].iloc[0]} \nPredicted Price: {predicted_price}"
    # )
    # print(analyzed_results)
    #
    # lower_bound, upper_bound = predictor.bootstrap_prediction_range(
    #     analyzed_results, predicted_price
    # )
    # classification_accuracy = predictor.evaluate_directional_accuracy(
    #     analyzed_results
    # )
    #

    # schema = etf_arg.lower()
    # model_id = predictor.save_model_details(schema=schema, features=endpoints)
    # predictor.save_model_predictions(
    #     schema=schema,
    #     model_id=model_id,
    #     prediction_dataframe=prediction_df,
    # )
    #
    # predictor.save_backtest_results(
    #     schema=schema,
    #     model_id=model_id,
    #     mape=calculated_pmae,
    #     cap=classification_accuracy,
    #     training_windows=backtest_predictor.n_windows,
    #     bootstrap_range=f"{lower_bound}-{upper_bound}",
    #     mpae_range=f"{prediction_range[0]}-{prediction_range[1]}",
    #     results_df=analyzed_results,
    # )
    #
    # predictor.save_data(
    #     connector=db_connector,
    #     model_id=model_id,
    #     df=train_data,
    #     table_name="training_data",
    #     schema=schema,
    # )
    #
    # predictor.save_data(
    #     connector=db_connector,
    #     model_id=model_id,
    #     df=predict_data,
    #     table_name="prediction_data",
    #     schema=schema,
    # )
