import pandas as pd 
import polars as pl
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as plt
import seaborn as sns
import re


def calculate_market_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate stock data into market-level features for each minute.
    
    Parameters:
    - df: Polars DataFrame containing stock data with necessary columns.
    
    Returns:
    - market_data: Polars DataFrame with aggregated market-level features.
    """
    market_data = (
        df.group_by('minute')
        .agg([
            # Price volatility (standard deviation of weighted trade prices)
            pl.col('weighted_trade_price').std().alias('price_volatility'),

            # Return volatility (volume-weighted standard deviation of returns)
            (
                (pl.col('return') * pl.col('total_trade_volume'))
                .sum()
                / pl.col('total_trade_volume').sum()
            ).alias('avg_return'),
            pl.col('return').std().alias('return_volatility'),

            

            # Trade volume statistics
            pl.col('total_trade_volume').mean().alias('avg_trade_volume'),
            pl.col('total_trade_volume').std().alias('volume_volatility'),

            # Average bid-ask spread
            ((pl.col('weighted_ask_price') - pl.col('weighted_bid_price')).mean())
            .alias('avg_spread'),

            # Proportion of advancing stocks (positive returns)
            (pl.col('return') > 0).mean().alias('advancing_stocks'),
        ])
        .fill_null(0)  # Replace null values with 0
    )

    # Ensure the results are sorted by minute
    market_data = market_data.sort('minute')

    return market_data


def create_windowed_clustering_polars(df, window=390, threshold=0.5):
    """
    Perform clustering within each rolling window, compute cluster characteristics,
    and suggest a trading strategy for the last element in the window.
    Also, compute the average market data within each window and ensure a valid cluster for the last element.

    Parameters:
        df (pl.DataFrame): Input data containing market features.
        window (int): Size of the rolling window.
        threshold (float): Minimum similarity value for the sparse similarity matrix.

    Returns:
        pl.DataFrame: Dataframe containing cluster characteristics, average market data,
                      transition matrices, last element's cluster, and trading strategies for each window.
    """
    results = []

    for start in range(len(df) - window + 1):
        # Define the rolling window
        window_df = df.slice(start, window)

        # Extract features for clustering
        clustering_features = [
            'price_volatility',
            'return_volatility',
            'avg_return',
            'avg_trade_volume',
            'volume_volatility',
            'advancing_stocks',
            'avg_spread'
        ]
        features = window_df.select(clustering_features).to_numpy()

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute similarity matrix
        similarity = cosine_similarity(features_scaled)
        np.fill_diagonal(similarity, 0)  # Avoid self-loops

        # Create graph based on the similarity matrix
        G = nx.Graph()
        for i in range(len(features_scaled)):
            G.add_node(i)  # Add all nodes, even if no edges exist

        for i in range(len(similarity)):
            for j in range(i + 1, len(similarity)):
                if similarity[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity[i, j])

        # Perform Louvain clustering on the graph
        partition = community_louvain.best_partition(G, weight='weight')

        # Ensure all nodes have a cluster
        for node in G.nodes:
            if node not in partition:
                # Assign to the nearest cluster
                closest_node = max(
                    ((other_node, similarity[node, other_node]) for other_node in G.nodes if node != other_node),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0]
                if closest_node is not None:
                    partition[node] = partition[closest_node]

        # Map clusters to the window DataFrame
        cluster_assignments = [partition[i] for i in range(len(window_df))]
        window_df = window_df.with_columns(pl.Series("cluster", cluster_assignments))

        # Compute cluster characteristics
        cluster_characteristics = []
        for cluster_id in set(partition.values()):
            cluster_data = window_df.filter(pl.col("cluster") == cluster_id)
            stats = {
                'cluster': cluster_id,
                'mean_return': cluster_data.select(pl.col('avg_return')).mean()[0, 0],
                'std_return': cluster_data.select(pl.col('avg_return')).std()[0, 0],
                'mean_volatility': cluster_data.select(pl.col('return_volatility')).mean()[0, 0],
                'avg_trade_volume': cluster_data.select(pl.col('avg_trade_volume')).mean()[0, 0],
                'avg_spread': cluster_data.select(pl.col('avg_spread')).mean()[0, 0],
                'size': cluster_data.height,
            }
            cluster_characteristics.append(stats)

        # Determine the cluster of the last element in the window
        last_element_cluster = window_df[-1, "cluster"]
        last_cluster_stats = next(
            (char for char in cluster_characteristics if char['cluster'] == last_element_cluster),
            None
        )

        # Compute average market data for the window
        avg_market_data = {
            'avg_price_volatility': window_df.select(pl.col('price_volatility')).mean()[0, 0],
            'avg_return_volatility': window_df.select(pl.col('return_volatility')).mean()[0, 0],
            'avg_return': window_df.select(pl.col('avg_return')).mean()[0, 0],
            'avg_trade_volume': window_df.select(pl.col('avg_trade_volume')).mean()[0, 0],
            'avg_advancing_stocks': window_df.select(pl.col('advancing_stocks')).mean()[0, 0],
            'avg_spread': window_df.select(pl.col('avg_spread')).mean()[0, 0],
        }

        # Store results for the window
        results.append({
            'start_time': window_df[0, "minute"],
            'end_time': window_df[-1, "minute"],
            'cluster_assignments': window_df.select(['minute', 'cluster']).to_dicts(),
            'cluster_characteristics': cluster_characteristics,
            'average_market_data': avg_market_data,
            'last_element_cluster': last_element_cluster
        })

    # Convert results into a Polars DataFrame
    final_df = pl.DataFrame(results)
    return final_df


def parse_cluster_data(cluster_assignments_str):
    # Clean up the string and add missing commas
    cleaned_str = cluster_assignments_str.strip('[]')
    cleaned_str = re.sub(r'}\s*{', '}, {', cleaned_str)
    cluster_data = eval('[' + cleaned_str + ']')
    return pd.DataFrame(cluster_data)



def plot_market_metrics(market_df, cluster_assignments):
    # Set the style
    

    
    # Create a figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle('Market Metrics Over Time by Cluster', fontsize=16, y=0.95)
    
    # Create color palette for clusters
    n_clusters = len(set(cluster_assignments['cluster']))
    # Create color palette for clusters
    # Create color palette for clusters (Red, Green, Yellow, Blue)
    # Create color palette with mustard yellow
    colors = sns.color_palette(["red", "green", "#D4A017", "blue"])  # #D4A017 is mustard yellow



    
    # Convert timestamps to datetime
    cluster_assignments['minute'] = pd.to_datetime(cluster_assignments['minute'])
    market_df['minute'] = pd.to_datetime(market_df['minute'])
    
    # Merge market data with cluster assignments
    merged_data = pd.merge(market_df, cluster_assignments, on='minute', how='inner')
    
    print("Merged data shape:", merged_data.shape)
    print("Number of unique clusters:", merged_data['cluster'].nunique())
    
    # Plot Return vs Time
    sns.scatterplot(data=merged_data, 
                   x='minute',
                   y='avg_return',
                   hue='cluster',
                   palette=colors,
                   ax=axes[0])
    axes[0].set_title('Return vs Time')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot Volatility vs Time
    sns.scatterplot(data=merged_data,
                   x='minute',
                   y='return_volatility',
                   hue='cluster',
                   palette=colors,
                   ax=axes[1])
    axes[1].set_title('return Volatility vs Time')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Spread vs Time
    sns.scatterplot(data=merged_data,
                   x='minute',
                   y='avg_spread',
                   hue='cluster',
                   palette=colors,
                   ax=axes[2])
    axes[2].set_title('Spread vs Time')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Plot Volume vs Time
    sns.scatterplot(data=merged_data,
                   x='minute',
                   y='avg_trade_volume',
                   hue='cluster',
                   palette=colors,
                   ax=axes[3])
    axes[3].set_title('Trade Volume vs Time')
    axes[3].tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

def plot_market_metrics_main(windowed_results_pd, market_data_pd):
    # Parse cluster assignments
    cluster_df = parse_cluster_data(str(windowed_results_pd.iloc[6]['cluster_assignments'].tolist()))
    print("Parsed cluster data shape:", cluster_df.shape)
    
    # Ensure market_data has the right column names
    if 'minute' not in market_data_pd.columns:
        print("Available columns in market_data:", market_data_pd.columns)
        # If minute is the index, reset it to be a column
        market_data_pd.reset_index(inplace=True)
        market_data_pd.rename(columns={'index': 'minute'}, inplace=True)
    
    # Ensure numeric columns are float
    numeric_columns = ['price_volatility', 'avg_return', 'return_volatility', 
                      'avg_trade_volume', 'volume_volatility', 'avg_spread', 'advancing_stocks']
    for col in numeric_columns:
        if col in market_data_pd.columns:
            market_data_pd[col] = pd.to_numeric(market_data_pd[col], errors='coerce')
    
    # Create and show the plot
    fig = plot_market_metrics(market_data_pd, cluster_df)
    plt.show()
