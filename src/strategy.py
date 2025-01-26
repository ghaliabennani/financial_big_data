
import pandas as pd 
import ast
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data_for_strat(merged_data):
    """
    Fix formatting issues and ensure columns contain valid Python objects.
    """
    def fix_format(value):
        if isinstance(value, (list, dict, np.ndarray)):
            return value  # Already abb valid object
        try:
            # Replace newlines with commas for valid syntax
            formatted_value = value.replace("\n", ", ")
            return ast.literal_eval(formatted_value)
        except Exception as e:
            raise ValueError(f"Cannot parse the value: {value}. Error: {e}")
    
    # Apply to columns with potential formatting issues
    merged_data["cluster_characteristics"] = merged_data["cluster_characteristics"].apply(fix_format)
    merged_data["average_market_data"] = merged_data["average_market_data"].apply(fix_format)
    return merged_data


def apply_strategy(merged_data, q = 0.6):
    """
    Apply a trading strategy using the given market data and cluster characteristics.

    Parameters:
        merged_data (pd.DataFrame): Input DataFrame containing market data and cluster characteristics.

    Returns:
        pd.DataFrame: DataFrame with trading actions, positions, and cumulative returns.
    """
    trading_results = []
    current_position = 0  # Initial position (0: none, +1: long, -1: short)
    cumulative_return = 0  # Tracks cumulative strategy return

    for _, row in merged_data.iterrows():
        last_cluster = row["last_element_cluster"]
        
        # Use `cluster_characteristics` and `average_market_data` directly
        cluster_characteristics = row["cluster_characteristics"]
        market_data_avg = row["average_market_data"]

        # Retrieve cluster data for the last element's cluster
        cluster_data = next((char for char in cluster_characteristics if char["cluster"] == last_cluster), None)
        if not cluster_data:
            continue  # Skip if no cluster data is found

        # Extract metrics
        cluster_mean_return = cluster_data["mean_return"]
        cluster_mean_volatility = cluster_data["mean_volatility"]
        cluster_avg_spread = cluster_data["avg_spread"]

        market_mean_return = market_data_avg["avg_return"]
        market_mean_volatility = market_data_avg["avg_return_volatility"]
        market_avg_spread = market_data_avg["avg_spread"]

        # Strategy logic
        action = "Hold"  # Default action
        position_change = 0

        if current_position == 0:  # No position
            if cluster_mean_return > market_mean_return + q * market_mean_volatility and cluster_avg_spread < market_avg_spread:
                action = "Buy"
                position_change = 1
            elif cluster_mean_return < market_mean_return - q * market_mean_volatility and cluster_avg_spread < market_avg_spread:
                action = "Sell"
                position_change = -1
        elif current_position == 1:  # Long position
            if cluster_mean_return < market_mean_return - q * 1.5 * market_mean_volatility:
                action = "Sell to Short"
                position_change = -2
            elif cluster_avg_spread > market_avg_spread + 0.01:
                action = "Close Position"
                position_change = -1
        elif current_position == -1:  # Short position
            if cluster_mean_return > market_mean_return + q * 1.5 * market_mean_volatility:
                action = "Buy to Long"
                position_change = 2
            elif cluster_avg_spread > market_avg_spread + 0.01:
                action = "Close Position"
                position_change = 1

        # Calculate returns
        last_avg_return = row["avg_return"]
        strategy_return = current_position * last_avg_return
        cumulative_return += strategy_return

        # Update current position
        current_position += position_change

        # Store results
        trading_results.append({
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "action": action,
            "position": current_position,
            "strategy_return": strategy_return,
            "cumulative_return": cumulative_return,
            "last_element_cluster": last_cluster,
            "cluster_data": cluster_data,
            "market_data_avg": market_data_avg,
            "last_avg_return": last_avg_return
        })

    return pd.DataFrame(trading_results)


def plot_cumulative_returns(strategy_results, merged_data, save_path=None):
    """
    Plot the cumulative returns of the strategy versus the market's cumulative returns.

    Parameters:
        strategy_results (pd.DataFrame): DataFrame containing strategy cumulative returns.
        merged_data (pd.DataFrame): Original merged market data for calculating market cumulative returns.
        save_path (str, optional): File path to save the plot. If None, the plot will not be saved.
    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    # Calculate market cumulative return
    merged_data["market_cumulative_return"] = merged_data["avg_return"].cumsum()

    # Extract strategy cumulative return
    strategy_cumulative_return = strategy_results["cumulative_return"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_results["end_time"], strategy_cumulative_return, label="Strategy Cumulative Return", linewidth=2)
    ax.plot(merged_data["end_time"], merged_data["market_cumulative_return"], label="Market Cumulative Return", linestyle="--", linewidth=2)
    
    # Add labels, title, legend, and grid
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Return: Strategy vs Market")
    ax.legend()
    ax.grid()
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Return the figure
    return fig