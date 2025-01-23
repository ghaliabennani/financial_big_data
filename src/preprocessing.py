import polars as pl
import os
import pandas as pd
import numpy as np
import shutil

def process_stock_with_polars(stock_dir, stock):
    """Merge bbo and trade files for a single stock using Polars and perform aggregations."""
    bbo_dir = os.path.join(stock_dir, 'bbo')
    trade_dir = os.path.join(stock_dir, 'trade')

    # Ensure subdirectories exist
    if not os.path.exists(bbo_dir) or not os.path.exists(trade_dir):
        print(f"Skipping {stock_dir}: Missing 'bbo' or 'trade' subdirectories.")
        return

    # Get sorted list of Parquet files in both directories
    bbo_files = sorted([f for f in os.listdir(bbo_dir) if f.endswith(".parquet")])
    trade_files = sorted([f for f in os.listdir(trade_dir) if f.endswith(".parquet")])

    lazy_dfs = []

    for bbo_file, trade_file in zip(bbo_files, trade_files):
        bbo_path = os.path.join(bbo_dir, bbo_file)
        trade_path = os.path.join(trade_dir, trade_file)

        if os.path.isfile(bbo_path) and os.path.isfile(trade_path):
            # Load the Parquet files as Polars lazy DataFrames
            bbo_df = pl.read_parquet(bbo_path).lazy()
            trade_df = pl.read_parquet(trade_path).lazy()

            # Merge on xltime
            merged_df = bbo_df.join(trade_df, on="xltime", how="inner")



            # Ensure consistent types
            merged_df = merged_df.with_columns([
                pl.col("xltime").cast(pl.Float64),
                pl.col("bid-price").cast(pl.Float64).alias("bid-price"),
                pl.col("bid-volume").cast(pl.Float64),
                pl.col("ask-price").cast(pl.Float64),
                pl.col("ask-volume").cast(pl.Float64),
                pl.col("trade-price").cast(pl.Float64),
                pl.col("trade-volume").cast(pl.Float64)
            ])

            # Perform initial aggregation by ⁠ xltime ⁠
            aggregated_df = (
                merged_df
                .group_by("xltime")
                .agg([
                    ((pl.col("bid-price") * pl.col("bid-volume")).sum() / pl.col("bid-volume").sum()).alias("weighted_bid_price"),
                    pl.col("bid-volume").sum().alias("total_bid_quantity"),
                    pl.col("bid-volume").max().alias("max_bid_quantity"),
                    ((pl.col("ask-price") * pl.col("ask-volume")).sum() / pl.col("ask-volume").sum()).alias("weighted_ask_price"),
                    pl.col("ask-volume").sum().alias("total_ask_quantity"),
                    pl.col("ask-volume").max().alias("max_ask_quantity"),
                    pl.col("trade-volume").sum().alias("total_trade_volume"),
                    ((pl.col("trade-price") * pl.col("trade-volume")).sum() / pl.col("trade-volume").sum()).alias("weighted_trade_price")
                ])
            )

            # Convert ⁠ xltime ⁠ to ⁠ datetime ⁠
            excel_base_date = pl.datetime(1899, 12, 30)
            aggregated_df = aggregated_df.with_columns(
                (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("datetime")
            )
            aggregated_df = aggregated_df.with_columns(pl.col("datetime").dt.convert_time_zone("America/New_York"))
            

            # Append the lazy DataFrame for processing
            lazy_dfs.append(aggregated_df)
        else:
            print(f"Skipping unmatched files: {bbo_file}, {trade_file}")

    if lazy_dfs:
        # Combine all daily lazy DataFrames into one
        all_data = pl.concat(lazy_dfs)

        # Collect and write to CSV
        final_df = all_data.collect()
        output_path = os.path.join(stock_dir, "merged_data_" +stock + ".csv")
        final_df.write_csv(output_path)
        print(f"Merged data saved as CSV for stock: {os.path.basename(stock_dir)}")
    else:
        print(f"No data to merge for stock: {os.path.basename(stock_dir)}")

def process_all_stocks_with_polars(base_dir):
    """Process all stocks in the base directory using Polars."""
    for stock in sorted(os.listdir(base_dir)):
        stock_dir = os.path.join(base_dir, stock)
        if os.path.isdir(stock_dir):
            print(f"Processing stock: {stock}")
            process_stock_with_polars(stock_dir, stock)
        else:
            print(f"Skipping non-directory item: {stock}")

def sort_csv_by_datetime(base_dir):
    """Sort each CSV file in the stock directories by datetime."""
    for stock in sorted(os.listdir(base_dir)):
        stock_dir = os.path.join(base_dir, stock)
        if os.path.isdir(stock_dir):
            csv_path = os.path.join(stock_dir, "merged_data_" + stock + ".csv")
            if os.path.exists(csv_path):
                try:
                    # Load the CSV file
                    df = pd.read_csv(csv_path)

                    # Ensure datetime column is in proper datetime format
                    df['datetime'] = pd.to_datetime(df['datetime'])

                    # Sort by datetime
                    df = df.sort_values(by='datetime')

                    # Save the sorted CSV back
                    df.to_csv(csv_path, index=False)
                    print(f"Sorted CSV saved for stock: {stock}")
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
            else:
                print(f"No CSV found for stock: {stock}")

def create_features(base_dir):
    """Process all stock data: aggregate by minute, calculate normalized returns."""
    for stock in sorted(os.listdir(base_dir)):
        stock_dir = os.path.join(base_dir, stock)
        csv_path = os.path.join(stock_dir, f"merged_data_{stock}.csv")
        
        if os.path.isdir(stock_dir) and os.path.exists(csv_path):
            print(f"Processing data for stock: {stock}")
            
            try:
                # Load the CSV file
                df = pd.read_csv(csv_path)

                # Ensure datetime column is properly formatted
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

                # Aggregate data by minute
                df['minute'] = df['datetime'].dt.floor('T')
                aggregated_df = (
                    df.groupby("minute")
                    .apply(
                        lambda group: pd.Series({
                            "weighted_bid_price": (group["weighted_bid_price"] * group["total_bid_quantity"]).sum() / group["total_bid_quantity"].sum(),
                            "max_bid_quantity": group["max_bid_quantity"].max(),
                            "total_bid_quantity": group["total_bid_quantity"].sum(),
                            "weighted_ask_price": (group["weighted_ask_price"] * group["total_ask_quantity"]).sum() / group["total_ask_quantity"].sum(),
                            "total_ask_quantity": group["total_ask_quantity"].sum(),
                            "max_ask_quantity": group["max_ask_quantity"].max(),
                            "weighted_trade_price": (group["weighted_trade_price"] * group["total_trade_volume"]).sum() / group["total_trade_volume"].sum(),
                            "total_trade_volume": group["total_trade_volume"].sum(),
                        })
                    )
                    .reset_index()
                )

                # Add columns for time differences and returns
                aggregated_df['date'] = aggregated_df['minute'].dt.date
                aggregated_df['delta_time'] = aggregated_df.groupby('date')['minute'].diff().dt.total_seconds() / 60
                aggregated_df['return'] = (
                    (aggregated_df['weighted_trade_price'] - aggregated_df['weighted_trade_price'].shift(1))
                    / aggregated_df['weighted_trade_price'].shift(1)
                )

                # Normalize returns by delta time
                aggregated_df['normalized_return'] = aggregated_df['return'] / aggregated_df['delta_time']
                aggregated_df['normalized_return'] = aggregated_df['normalized_return'].where(
                    aggregated_df['delta_time'] > 0, np.nan
                )

                # Save the processed DataFrame back
                output_path = os.path.join(stock_dir, f"processed_data_{stock}.csv")
                aggregated_df.to_csv(output_path, index=False)
                print(f"Processed data saved for stock: {stock}")
            except Exception as e:
                print(f"Error processing data for stock {stock}: {e}")
        else:
            print(f"Skipping {stock}: No valid data found.")


def create_processed_data_dir(base_dir):
    main_directory = base_dir

    # Define the target directory for processed data files
    target_directory = os.path.join(main_directory, 'processed_data')

    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # Loop through the subdirectories
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            # Construct the expected file name
            processed_file_name = f"processed_data_{subdir}.csv"
            processed_file_path = os.path.join(subdir_path, processed_file_name)
        
            # Check if the processed data file exists
            if os.path.exists(processed_file_path):
                # Move the file to the target directory
                shutil.move(processed_file_path, os.path.join(target_directory, processed_file_name))
                print(f"Moved: {processed_file_path} -> {target_directory}")
            else:
                print(f"File not found: {processed_file_path}")
    return target_directory


def filter_dataset(base_dir):
    # Path to the folder containing stock data files
    folder_path = base_dir

    # Define the start and end time
    start_time = pd.to_datetime('09:30:00').time()
    end_time = pd.to_datetime('16:00:00').time()

    filtered_dataframes = {}

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Assuming files are in CSV format
            # Construct the file path
            file_path = os.path.join(folder_path, file_name)

            # Load the stock data
            stock_df = pd.read_csv(file_path)

            # Convert the 'minute' column to datetime
            stock_df['minute'] = pd.to_datetime(stock_df['minute'])

            # Filter rows between the specified times
            df_filtered = stock_df[
                (stock_df['minute'].dt.time >= start_time) &
                (stock_df['minute'].dt.time <= end_time)&
                (stock_df['weighted_ask_price'] > stock_df['weighted_bid_price']) &
                (stock_df['weighted_ask_price'] > 0) &
                (stock_df['weighted_bid_price'] > 0)
            ]

            # Store the filtered dataframe in a dictionary
            filtered_dataframes[file_name] = df_filtered

            # Save the filtered data back to a new file
            output_file_name = f"{file_name}"
            output_path = os.path.join(folder_path, output_file_name)
            df_filtered.to_csv(output_path, index=False)

    # Display a summary message
    print(f"Processed and saved filtered data for {len(filtered_dataframes)} files.")


def create_merged_dataset(base_dir):
    # Path to the folder containing stock data files
    folder_path = base_dir

    # List to store individual DataFrames
    all_dfs = []

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            # Load the stock data
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            
            # Extract stock name from the file name
            stock_name = file_name.split("_")[-1].replace(".csv", "")
            
            # Add a column identifying the stock
            df['stock'] = stock_name
            
            # Append to the list
            all_dfs.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save the combined DataFrame to disk
    combined_df.to_csv( base_dir+ '/combined_stock_data.csv', index=False)