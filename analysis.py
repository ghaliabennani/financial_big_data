from . import clustering
import src
from src import preprocessing, strategy;
from src.preprocessing import *
from clustering import *
from src.strategy import *
import argparse
import warnings
warnings.filterwarnings('ignore')


def full_analysis():
    print("Running full analysis on the entire dataset...")
    return '/Users/ghaliabennani/Desktop/financial_big_data/data_reduced'

def demo_workflow():
    print("Running demo workflow on a subset of the data...")
    return '/Users/ghaliabennani/Desktop/financial_big_data/data_reduced'


def main():
    #include the --full argument to run the full analysis

    parser = argparse.ArgumentParser(description="Control whether to run full analysis or demo workflow.")
    parser.add_argument(
        "--full", 
        action="store_true",  # Sets this flag to True if provided
        help="Run the full analysis (default is demo workflow)."
    )
    args = parser.parse_args()
    
    # Control the workflow based on the switch
    data_dir = ''
    if args.full:
        data_dir = full_analysis()
    else:
        data_dir = demo_workflow()

    #First manipulation and preprocessing of the data 
    process_all_stocks_with_polars(data_dir)
    sort_csv_by_datetime(data_dir)
    create_features(data_dir)
    processed_data_path = create_processed_data_dir(data_dir)
    filter_dataset(processed_data_path)
    create_merged_dataset(processed_data_path)


    #Clustering
    combined_df = pd.read_csv(processed_data_path + '/combined_stock_data.csv')
    combined_df = pl.DataFrame(combined_df)
    market_data = calculate_market_data(combined_df)
    windowed_results = create_windowed_clustering_polars(market_data, window=780, threshold=0.5)
    windowed_results.to_csv(data_dir + '/windowed_results.csv', index=False)


    #some preprocessing before applying the strategy
    market_data_pd = market_data.to_pandas()
    windowed_results_pd = windowed_results.to_pandas()
    merged_data = pd.merge(windowed_results_pd, market_data_pd, left_on="end_time", right_on="minute", how="inner")
    merged_data= merged_data.drop(columns=['minute'])
    merged_data = preprocess_data_for_strat(merged_data)

    #Apply the strategy
    strategy_results = apply_strategy(merged_data)
    strategy_results.to_csv(data_dir + '/strategy_results.csv', index=False)

    #Some vizualizations
    fig1 = plot_cumulative_returns(strategy_results, merged_data)
    fig1.savefig("market_metrics_by_cluster.png", dpi=300, bbox_inches='tight')
    fig2 = plot_market_metrics_main(windowed_results_pd, market_data_pd)
    fig2.savefig("market_metrics_by_cluster.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()