"""
Latency Arbitrage - Core Functions Module

This module contains the core trading functions for latency arbitrage.
It handles position sizing, slippage modeling, backtest execution, 
statistical analysis, and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from collections import deque
from statsmodels.stats.power import TTestPower
from scipy import stats
from typing import List, Dict, Tuple, Any, Optional
from scipy.stats import ttest_1samp

# ===== 1. POSITION SIZING =====

def half_kelly(
    trade_size: float,
    capital: float,
    entry_price: float,
    p: float = 0.75,
    b: float = 0.1,
    max_fraction: float = 0.5,
    min_qty: float = 0.0,
    max_qty: float = 1_000_000.0,
    recent_volatility: float = 1.0,
) -> float:
    """
    Computes a half-Kelly fraction for dynamic sizing, optionally weighting by recent volatility
    and factoring in risk adjustments.

    half-Kelly fraction formula: f = 0.5 * ((p*(b+1) - 1) / b)

    Then we scale by:
      - 'trade_size' (the magnitude of the triggering trade, if desired)
      - 1 / recent_volatility  (if volatility is higher, we reduce size)
      - drawdown_adjust_factor (less than 1 if we want to shrink after a drawdown)

    The resulting fraction is then clamped by [0, max_fraction].
    Then we convert to a coin quantity => fraction_of_capital * (capital / entry_price),
    and clamp to [min_qty, max_qty].
    """
    raw_kelly = 0.5 * ((p*(b+1) - 1) / b)
    if raw_kelly <= 0:
        return 0.0

    # Scale by trade_size
    scale_factor = trade_size

    # Adjust for volatility and drawdown factor
    # If volatility is large, fraction_of_capital gets smaller
    # If drawdown_adjust_factor < 1, it also shrinks
    scaled_kelly = raw_kelly * scale_factor * (1.0 / max(recent_volatility, 1e-8))

    fraction_of_capital = min(scaled_kelly, max_fraction)

    if entry_price <= 0:
        return 0.0

    raw_order_qty = fraction_of_capital * (capital / entry_price)
    order_qty = max(min_qty, min(raw_order_qty, max_qty))
    return order_qty


# ===== 2. REALISTIC SLIPPAGE MODELING =====

def simulate_dynamic_market_impact(
    book_levels: List[Tuple[float, float]],
    concurrency_factor: float
) -> List[Tuple[float, float]]:
    """
    Optionally reduce or shift liquidity further to model concurrency/market impact.

    This function simulates the effect of other traders ALSO hitting the same levels in parallel,
    leading to even less liquidity than naive slippage might imply.

    concurrency_factor in [0, 1]:
      0 => no concurrency, 1 => extremely heavy concurrency that removes nearly all size.

    We reduce each level's size by concurrency_factor * size. (Simple approach)
    """
    if concurrency_factor <= 0:
        return book_levels

    new_levels = []
    for (price, size) in book_levels:
        concurrency_depletion = concurrency_factor * size
        new_size = max(0, size - concurrency_depletion)
        new_levels.append((price, new_size))

    return new_levels

def simulate_fill_price(
    order_qty: float,
    book_levels: List[Tuple[float, float]],
    direction: int,
    latency_ms: float = 0,
    concurrency_factor: float = 0.0
) -> Tuple[Optional[float], float]:
    """
    Simulates realistic fill price by walking the order book, with optional concurrency modeling.
    
    We allow partial fills if not enough liquidity is at these levels. The leftover qty is not filled.
    """
    qty_remaining = order_qty
    total_cost = 0.0
    filled_qty = 0.0

    # Copy original levels
    levels_copy = book_levels.copy()

    # 1) Simulate concurrency-based depletion
    if concurrency_factor > 0:
        levels_copy = simulate_dynamic_market_impact(levels_copy, direction, concurrency_factor)

    # 2) Latency-based depletion: simple approach => reduce each level's size
    if latency_ms > 0:
        # We treat latency_ms as a fraction that depletes up to 90% of liquidity
        decay_factor = min(0.9, latency_ms / 1000.0)
        for i, (price, size) in enumerate(levels_copy):
            # Example: deeper levels are less impacted, so multiply by (1 - i / len(levels_copy))
            depth_multiplier = 1.0 - (i / max(len(levels_copy), 1))
            level_decay = decay_factor * depth_multiplier
            new_size = size * (1.0 - level_decay)
            levels_copy[i] = (price, max(new_size, 0.0))

    # 3) Walk the possibly-depleted book to fill
    for price, available in levels_copy:
        if available <= 0:
            continue
        fill_qty = min(qty_remaining, available)
        total_cost += fill_qty * price
        filled_qty += fill_qty
        qty_remaining -= fill_qty

        if qty_remaining <= 0:
            break

    if filled_qty == 0:
        return None, 0.0

    vwap = total_cost / filled_qty
    return vwap, filled_qty


def sample_latency(mean_latency_ms: float, std_dev_ms: float) -> float:
    """
    Samples a latency value from a lognormal distribution.
    """
    mu = np.log(max(mean_latency_ms, 1e-6))
    sigma = np.log(1 + max(std_dev_ms, 1e-6) / max(mean_latency_ms, 1e-6))
    return float(np.random.lognormal(mu, sigma))


def extract_book_levels(
    row: pd.Series,
    depth: int = 5,
    include_full_levels: bool = False
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Extracts synthetic top bids/asks from row's best quotes, or from 'levels' if available.
    """
    if 'BidPrice_1' in row and 'BidSize_1' in row:
        bid_levels, ask_levels = [], []
        bid_px = row['BidPrice_1']
        bid_sz = row['BidSize_1']
        ask_px = row['AskPrice_1']
        ask_sz = row['AskSize_1']
        for i in range(depth):
            decay = 1.0 - (i * 0.1)
            # SHIFT the price up/down slightly for each deeper level
            bid_price_i = max(0.0, bid_px - (i * 0.0001 * bid_px))
            ask_price_i = ask_px + (i * 0.0001 * ask_px)

            bid_size_i = max(0.0, bid_sz * decay)
            ask_size_i = max(0.0, ask_sz * decay)
            bid_levels.append((bid_price_i, bid_size_i))
            ask_levels.append((ask_price_i, ask_size_i))
        return bid_levels, ask_levels

    if include_full_levels and 'levels' in row:
        try:
            levels = row['levels']
            # levels[0] => bids, levels[1] => asks
            bids = [(lvl['px'], lvl['sz']) for lvl in levels[0][:depth]]
            asks = [(lvl['px'], lvl['sz']) for lvl in levels[1][:depth]]
            return bids, asks
        except (IndexError, KeyError, TypeError):
            return [], []
    return [], []


# ===== 3. STATISTICAL SIGNIFICANCE TESTING =====

def bootstrap_returns(
    returns: np.ndarray,
    iterations: int = 1000,
    confidence: float = 0.95,
    block_size: int = 5
) -> Tuple[float, Tuple[float, float]]:
    """
    Block bootstrap for returns, preserving some autocorrelation.
    """
    n = len(returns)
    if n < 10:
        return np.mean(returns), (np.nan, np.nan)

    boot_means = []
    original_mean = np.mean(returns)

    for _ in range(iterations):
        resampled = []
        while len(resampled) < n:
            start_idx = np.random.randint(0, n - block_size + 1)
            block = returns[start_idx:start_idx+block_size]
            resampled.extend(block)
        resampled = resampled[:n]
        boot_means.append(np.mean(resampled))

    alpha = (1 - confidence) / 2
    lower_ci, upper_ci = np.percentile(boot_means, [100*alpha, 100*(1 - alpha)])

    return original_mean, (lower_ci, upper_ci)


def calculate_power_sample_size(
    returns: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.8
) -> float:
    """
    Required sample size (Cohen's d).
    """
    if len(returns) < 10:
        return np.nan
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return np.nan
    effect_size = abs(mean_return) / std_return

    analysis = TTestPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        alternative='two-sided'
    )
    return sample_size


def calculate_annualized_sharpe(
    returns: np.ndarray,
    timestamps: Optional[np.ndarray] = None
) -> float:
    """
    Annualized Sharpe from per-trade returns, if timestamps provided.
    """
    if len(returns) < 2:
        return np.nan

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return np.nan

    per_trade_sharpe = mean_return / std_return

    if timestamps is not None and len(timestamps) >= 2:
        sorted_timestamps = np.sort(timestamps)
        start_date = sorted_timestamps[0]
        end_date = sorted_timestamps[-1]

        if isinstance(start_date, np.datetime64):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, np.datetime64):
            end_date = pd.Timestamp(end_date)

        days = (end_date - start_date).days
        years = days / 365.25
        if years > 0:
            trades_per_year = len(returns) / years
            return per_trade_sharpe * np.sqrt(trades_per_year)

    return per_trade_sharpe


def check_signal_significance(trade_profits: np.ndarray, capital: float, alpha: float = 0.05) -> dict:
    """
    Checks whether the signal (i.e., the trades generated by the extreme quantile threshold)
    is statistically significant. This uses a one-sample one-sided t-test on the per-trade returns.
    
    Hypotheses:
      H0: μ = 0  (no exploitable edge; mean return is zero)
      H1: μ > 0  (positive edge; mean return is significantly positive)
    """
    # Compute per-trade returns
    returns = trade_profits / capital
    n = len(returns)
    if n < 2:
        return {
            'sample_size': n,
            'mean_return': np.nan,
            'std_return': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False
        }
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # Two-sided t-test
    t_stat, p_two = stats.ttest_1samp(returns, popmean=0.0)
    # Convert to one-sided p-value. We want to test if mean > 0.
    if mean_return > 0:
        p_one = p_two / 2.0
    else:
        p_one = 1.0 - p_two / 2.0
    significant = (p_one < alpha)
    
    return {
        'sample_size': n,
        'mean_return': mean_return,
        'std_return': std_return,
        't_statistic': t_stat,
        'p_value': p_one,
        'significant': significant
    }


# === Risk Metrics (Drawdown & Sortino) ===

def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Computes the maximum drawdown given an array of equity values over time.
    Each element in equity_curve is the total account value at that point.
    
    Returns:
        max_dd (float): The largest percentage drawdown, e.g. 0.20 = 20% drawdown.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for val in equity_curve:
        if val > peak:
            peak = val
        # Drawdown fraction = (peak - current) / peak
        dd = (peak - val) / peak if peak > 1e-9 else 0.0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def compute_sortino_ratio(
    returns: np.ndarray,
    rf: float = 0.0
) -> float:
    """
    Computes the Sortino ratio = (mean(returns) - rf) / std(negative returns).
    """
    if len(returns) < 2:
        return np.nan
    downside = returns[returns < rf]
    if len(downside) < 2:
        return np.nan
    mean_ret = np.mean(returns) - rf
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return np.nan
    return mean_ret / downside_std


# ===== 4. BACKTEST CORE COMPONENTS =====

def calculate_slippage_and_entry_price(
    aligned_index: int,
    direction: int,
    base_entry_price: float,
    trade_size: float,
    lob_data: pd.DataFrame,
    book_depth: int,
    latency: float,
    slippage_factor: float,
    current_concurrency_factor: float
) -> Tuple[float, float]:
    """
    Calculate slippage and determine entry price.
    """
    slippage_entry = 0.0

    try:
        lob_row = lob_data.iloc[aligned_index]
        bid_levels, ask_levels = extract_book_levels(lob_row, depth=book_depth)

        if direction > 0:
            entry_levels = ask_levels
        else:
            entry_levels = bid_levels

        if entry_levels:
            vwap, filled_qty = simulate_fill_price(
                trade_size,
                entry_levels,
                direction,
                latency_ms=latency * slippage_factor,
                concurrency_factor=current_concurrency_factor
            )
            if vwap is None or filled_qty <= 0:
                return None, 0.0
                
            entry_price = vwap
            slippage_entry = abs(entry_price - base_entry_price) / base_entry_price
        else:
            entry_price = base_entry_price
    except (IndexError, KeyError):
        entry_price = base_entry_price
        
    return entry_price, slippage_entry


def create_trade_result(
    trade_type: str,
    profit: float,
    entry_price: float,
    exit_price: float,
    entry_slippage: float,
    exit_slippage: float,
    order_quantity: float,
    aligned_index: int,
    exit_index: int,
    latency_ms: float,
    timestamp,
    direction: int,
    imbalance: float = None,
    entry_fee: float = None,
    exit_fee: float = None,
    detailed_logging: bool = False
) -> Dict[str, Any]:
    """
    Creates a standardized trade result dictionary.
    """
    trade_result = {
        'type': trade_type,
        'profit': profit,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'entry_slippage': entry_slippage,
        'exit_slippage': exit_slippage,
        'order_quantity': order_quantity,
        'aligned_index': aligned_index,
        'exit_index': exit_index,
        'holding_period': exit_index - aligned_index,
        'latency_ms': latency_ms,
        'timestamp': timestamp,
        'direction': direction
    }
    
    if detailed_logging:
        trade_result.update({
            'imbalance': imbalance,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee
        })
        
    return trade_result


def compile_backtest_results(
    total_profit: float,
    successful_trades: int,
    failed_trades: int,
    skipped_trades: int,
    total_slippage_entry: float,
    total_slippage_exit: float,
    slippage_count: int,
    trade_results: List[Dict[str, Any]],
    capital: float
) -> Dict[str, Any]:
    """
    Compiles backtest results into a summary dictionary.
    Now we base max drawdown on actual equity, not raw PnL from zero.
    """
    total_executed = successful_trades + failed_trades
    win_rate = (successful_trades / total_executed) if total_executed else 0.0
    avg_slippage_entry = total_slippage_entry / slippage_count if slippage_count > 0 else 0
    avg_slippage_exit = total_slippage_exit / slippage_count if slippage_count > 0 else 0

    summary = {
        'Total Profit': total_profit,
        'Successful Trades': successful_trades,
        'Failed Trades': failed_trades,
        'Win Rate': win_rate,
        'Skipped': skipped_trades,
        'Avg Entry Slippage': avg_slippage_entry,
        'Avg Exit Slippage': avg_slippage_exit
    }

    if trade_results:
        trade_df = pd.DataFrame(trade_results)
        
        # Some summary stats
        summary['Avg Profit Per Trade'] = trade_df['profit'].mean()
        summary['Profit Std Dev'] = trade_df['profit'].std()
        summary['Max Profit'] = trade_df['profit'].max()
        summary['Min Profit'] = trade_df['profit'].min()
        summary['Total Trades'] = len(trade_df)
        
        # 1) Instead of 'cumulative_profit', we build an equity column
        # 2) The equity starts at 'capital' + cumulative PnL
        trade_df['equity'] = capital + trade_df['profit'].cumsum()

        # 3) Now compute max drawdown from equity
        max_dd = compute_max_drawdown(trade_df['equity'].values)
        summary['Max Drawdown'] = max_dd
        
        # Optional: if you want to display max drawdown in percentage:
        # summary['Max Drawdown %'] = max_dd * 100.0

        # Sharpe & Sortino from per-trade returns
        returns = trade_df['profit'] / capital
        mean_return = returns.mean()
        std_return = returns.std(ddof=1)
        if std_return > 1e-15:
            summary['Sharpe Ratio'] = mean_return / std_return
            
            # Optionally compute annualized Sharpe if timestamps exist
            if 'timestamp' in trade_df.columns and trade_df['timestamp'].notna().sum() >= 2:
                ann_sharpe = calculate_annualized_sharpe(returns.values, trade_df['timestamp'].values)
                summary['Annualized Sharpe'] = ann_sharpe
        
        # Sortino ratio
        from math import sqrt
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std(ddof=1)
            if downside_std > 1e-15:
                sortino = (mean_return - 0.0) / downside_std
                summary['Sortino Ratio'] = sortino

    return summary

def check_imbalance_signal(
    aligned_index: int,
    direction: int,
    imbalance_array: np.ndarray,
    imb_threshold_long: float,
    imb_threshold_short: float
) -> bool:
    """
    Check if order book imbalance confirms the trade direction.
    
    Args:
        aligned_index: Current index in the imbalance array
        direction: Trade direction (1 for long, -1 for short)
        imbalance_array: Array of order book imbalance values
        imb_threshold_long: Threshold for long entries (negative value)
        imb_threshold_short: Threshold for short entries (positive value)
    
    Returns:
        bool: True if imbalance confirms direction, False otherwise
    """
    if aligned_index >= len(imbalance_array):
        return False
        
    current_imbalance = imbalance_array[aligned_index]
    if np.isnan(current_imbalance):
        return False
        
    # For longs: imbalance should be below long threshold (more bids than asks)
    if direction > 0:
        return current_imbalance <= imb_threshold_long
    # For shorts: imbalance should be above short threshold (more asks than bids)
    else:
        return current_imbalance >= imb_threshold_short


def determine_trade_direction(
    trade_row,
    aligned_index: int, 
    imbalance_array: np.ndarray,
    bid_price_array: np.ndarray, 
    ask_price_array: np.ndarray,
    mid_price_array: np.ndarray,
    microprice_array: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[Optional[int], Optional[float], Optional[str], float, float]:
    """
    Determine trade direction using imbalance and microprice confirmation.
    
    Args:
        trade_row: Row from trades dataframe
        aligned_index: Index in LOB data corresponding to this trade
        imbalance_array: Order book imbalance values
        bid_price_array: Best bid prices
        ask_price_array: Best ask prices
        mid_price_array: Midprice values
        microprice_array: Microprice values
        params: Dictionary of parameters
    
    Returns:
        Tuple containing:
            - direction (1=long, -1=short, None=no trade)
            - base entry price (None if no trade)
            - exit price type ('BidPrice_1' or 'AskPrice_1', None if no trade)
            - current imbalance value
            - signal strength (0-1)
    """
    # Extract parameters
    imb_threshold_long = params.get('imb_threshold_long', -0.3)
    imb_threshold_short = params.get('imb_threshold_short', 0.3)
    use_microprice = params.get('use_microprice', True)
    
    # Get basic direction from trade
    is_buyer_maker = trade_row.is_buyer_maker
    current_imbalance = imbalance_array[aligned_index] if aligned_index < len(imbalance_array) else 0.0
    
    if is_buyer_maker:
        # SELL => short
        direction = -1
        exit_price_type = 'AskPrice_1'
        base_entry_price = bid_price_array[aligned_index] if aligned_index < len(bid_price_array) else None
    else:
        # BUY => long
        direction = 1
        exit_price_type = 'BidPrice_1'
        base_entry_price = ask_price_array[aligned_index] if aligned_index < len(ask_price_array) else None
    
    # Check if entry price is valid
    if base_entry_price is None or np.isnan(base_entry_price):
        return None, None, None, current_imbalance, 0.0
    
    # Check imbalance signal
    imbalance_confirms = check_imbalance_signal(
        aligned_index, 
        direction, 
        imbalance_array, 
        imb_threshold_long, 
        imb_threshold_short
    )
    
    # Check microprice confirmation
    microprice_confirms = True
    if use_microprice and aligned_index < len(mid_price_array) and aligned_index < len(microprice_array):
        current_mid = mid_price_array[aligned_index]
        current_micro = microprice_array[aligned_index]
        
        if not np.isnan(current_mid) and not np.isnan(current_micro):
            # For longs: need microprice > midprice (buying pressure)
            # For shorts: need microprice < midprice (selling pressure)
            if direction > 0:  # Long signal
                microprice_confirms = current_micro >= current_mid
            else:  # Short signal
                microprice_confirms = current_micro <= current_mid
    
    # Only proceed if all required signals confirm
    if not (imbalance_confirms and microprice_confirms):
        return None, None, None, current_imbalance, 0.0
    
    # Calculate signal strength (0-1) based on how far imbalance exceeds threshold
    if direction > 0:
        # For longs, stronger signal when imbalance is more negative than threshold
        signal_strength = min(1.0, abs(current_imbalance / imb_threshold_long)) if imb_threshold_long != 0 else 0.5
    else:
        # For shorts, stronger signal when imbalance is more positive than threshold
        signal_strength = min(1.0, current_imbalance / imb_threshold_short) if imb_threshold_short != 0 else 0.5
    
    return direction, base_entry_price, exit_price_type, current_imbalance, signal_strength


# ===== 2. POSITION SIZING =====

def conservative_position_sizing(
    capital: float,
    entry_price: float,
    signal_strength: float = 0.5,
    max_risk_pct: float = 0.005,
    min_qty: float = 0.0,
    max_qty: float = 1000.0,
    recent_volatility: float = 1.0,
) -> float:
    """
    More conservative position sizing that scales with signal strength and limits risk.
    
    Args:
        trade_size: Size of the observed trade
        capital: Trading capital
        entry_price: Entry price
        signal_strength: Strength of the signal (0-1)
        max_risk_pct: Maximum percentage of capital to risk per trade
        min_qty: Minimum position size
        max_qty: Maximum position size
        recent_volatility: Recent market volatility measure
        drawdown_adjust_factor: Factor to reduce size after drawdowns
    
    Returns:
        float: Position size in quantity
    """
    # Base position size as percentage of capital
    max_position_value = capital * max_risk_pct
    
    # Scale by signal strength (0-1), volatility, and drawdown
    adjusted_position = max_position_value * signal_strength * (1.0 / max(recent_volatility, 1e-8))
    
    # Convert to quantity
    if entry_price <= 0:
        return 0.0
        
    raw_qty = adjusted_position / entry_price
    
    # Ensure within limits
    order_qty = max(min_qty, min(raw_qty, max_qty))
    return order_qty



# ===== 3. EXIT MANAGEMENT =====

def check_trailing_stop(
    current_price: float,
    best_price: float,
    direction: int,
    trailing_distance: float
) -> bool:
    """
    Check if trailing stop should be triggered.
    
    Args:
        current_price: Current market price
        best_price: Best price seen since trade entry
        direction: Trade direction (1=long, -1=short)
        trailing_distance: Distance for trailing stop as fraction
    
    Returns:
        bool: True if trailing stop should be triggered
    """
    if direction > 0:  # Long
        return current_price < (best_price * (1 - trailing_distance))
    else:  # Short
        return current_price > (best_price * (1 + trailing_distance))

def update_best_price(
    current_price: float,
    best_price: float,
    direction: int
) -> float:
    """
    Update the best price seen during the trade.
    
    Args:
        current_price: Current market price
        best_price: Previous best price
        direction: Trade direction (1=long, -1=short)
    
    Returns:
        float: Updated best price
    """
    if direction > 0:  # Long - track highest price
        return max(current_price, best_price)
    else:  # Short - track lowest price
        return min(current_price, best_price)

def process_trade_exits(
    direction: int,
    base_exit_price: float,
    average_entry_price: float,
    position_remaining: float,
    entry_fee: float,
    best_price_seen: float,
    trailing_stop_activated: bool,
    params: Dict[str, Any]
) -> Tuple[bool, str, float, float, float, bool, float]:
    """
    Process exit conditions (profit target and trailing stop) and determine if we should exit.
    
    Returns:
        A 7-element tuple:
            1) exit_triggered (bool)          - True if an exit condition is met
            2) exit_type (str)                - "profit_target" or "trailing_stop" ("" if no exit)
            3) realized_profit (float)        - net profit on this exit
            4) exit_price (float)             - the final exit price used
            5) updated_best_price (float)     - updated best/worst price for the trailing stop
            6) updated_trailing_activated (bool)
            7) exit_fee (float)               - taker fee for this exit
    """
    # Extract parameters
    taker_fee = params.get('taker_fee', 0.0003)
    fee_multiple = params.get('fee_multiple', 3.0)
    use_trailing_stop = params.get('use_trailing_stop', True)
    trailing_activation_bps = params.get('trailing_activation_bps', 3.0)
    trailing_distance_bps = params.get('trailing_distance_bps', 1.0)
    
    # Convert basis points to decimals
    trailing_activation = trailing_activation_bps / 10000.0
    trailing_distance = trailing_distance_bps / 10000.0
    
    # Calculate current P&L metrics
    exit_price = base_exit_price
    price_change = (exit_price - average_entry_price) * direction
    price_change_pct = price_change / average_entry_price
    exit_fee = exit_price * position_remaining * taker_fee
    
    # Update the best price seen for trailing-stop logic
    updated_best_price = update_best_price(exit_price, best_price_seen, direction)
    updated_trailing_activated = trailing_stop_activated

    # 1. Check profit target
    round_trip_fees = entry_fee + exit_fee
    net_pnl = price_change * position_remaining - entry_fee - exit_fee
    
    if net_pnl >= fee_multiple * round_trip_fees:
        # Profit target exit
        return True, 'profit_target', net_pnl, exit_price, updated_best_price, updated_trailing_activated, exit_fee
    
    # 2. Check trailing stop
    if use_trailing_stop:
        # Activate trailing stop if profit exceeds trailing_activation threshold (in %)
        if price_change_pct >= trailing_activation and not trailing_stop_activated:
            updated_trailing_activated = True
        
        # If trailing stop is activated, see if we have fallen back enough from the best price
        if updated_trailing_activated:
            if check_trailing_stop(exit_price, updated_best_price, direction, trailing_distance):
                realized_profit = price_change * position_remaining - entry_fee - exit_fee
                return True, 'trailing_stop', realized_profit, exit_price, updated_best_price, updated_trailing_activated, exit_fee
    
    # No exit condition met
    return False, '', 0.0, exit_price, updated_best_price, updated_trailing_activated, exit_fee


# ===== 4. MAIN TRADE PROCESSING =====

def process_trade(
    trade,
    lob_data: pd.DataFrame,
    BidPrice1_array: np.ndarray,
    AskPrice1_array: np.ndarray,
    Imbalance_array: np.ndarray,
    MidPrice_array: np.ndarray,
    Microprice_array: np.ndarray,
    n_rows: int,
    equity_curve: List[float],
    recent_volumes: deque,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, float, int, int, int]:
    """
    Process a single trade in the backtest with trailing-stop or profit-target exit.
    If no exit triggers by the final row, we perform a final 'mark-to-market' exit
    to realize the PnL at the end of the dataset.

    Returns:
        1) summary (Dict[str, Any]) - typically empty
        2) trade_results (List[Dict[str, Any]]) - the trade(s) for final exit
        3) profit (float) - realized profit on final exit
        4) successful_profit (float) - same if >0, else 0
        5) success_count (int) - 1 if final exit profit>0
        6) fail_count (int) - 1 if final exit profit<0
        7) skip_count (int) - 1 if trade was skipped, else 0
    """
    # ==============================
    # 1) Parse parameters
    # ==============================
    maker_fee = params.get('maker_fee', 0.0001)
    capital = params.get('capital', 10_000.0)
    min_size = params.get('min_size', 0.05)
    max_size = params.get('max_size', 0.2)
    concurrency_factor = params.get('concurrency_factor', 0.0)
    simulate_slippage = params.get('simulate_slippage', True)
    mean_latency_ms = params.get('mean_latency_ms', 50.0)
    std_dev_latency_ms = params.get('std_dev_latency_ms', 20.0)
    book_depth = params.get('book_depth', 5)
    slippage_factor = params.get('slippage_factor', 1.0)
    detailed_logging = params.get('detailed_logging', True)
    max_risk_pct = params.get('max_risk_pct', 0.002)

    aligned_index = trade.aligned_index
    trade_size = trade.qty
    trade_timestamp = getattr(trade, 'aligned_timestamp', None)

    # ==============================
    # 2) Determine direction with multiple confirmations
    # ==============================
    direction, base_entry_price, exit_price_type, current_imbalance, signal_strength = determine_trade_direction(
        trade,
        aligned_index,
        Imbalance_array,
        BidPrice1_array,
        AskPrice1_array,
        MidPrice_array,
        Microprice_array,
        params
    )
    
    # If no valid direction, skip this trade
    if direction is None:
        return {}, [], 0, 0, 0, 0, 1

    # ==============================
    # 3) Calculate entry price with slippage
    # ==============================
    entry_price = base_entry_price
    latency = 0.0
    slippage_entry = 0.0

    if simulate_slippage:
        latency = sample_latency(mean_latency_ms, std_dev_latency_ms)
        entry_price, slippage_entry = calculate_slippage_and_entry_price(
            aligned_index,
            direction,
            base_entry_price,
            trade_size,
            lob_data,
            book_depth,
            latency,
            slippage_factor,
            concurrency_factor
        )
        # If we cannot get filled at all, skip
        if entry_price is None:
            return {}, [], 0, 0, 0, 0, 1
    
    # ==============================
    # 4) Position sizing
    # ==============================
    if len(recent_volumes) > 10:
        recent_volatility = np.std(recent_volumes) + 1.0
    else:
        recent_volatility = 1.0

    order_qty = conservative_position_sizing(
        trade_size=trade_size,
        capital=capital,
        entry_price=entry_price,
        signal_strength=signal_strength,
        max_risk_pct=max_risk_pct,
        min_qty=min_size,
        max_qty=max_size,
        recent_volatility=recent_volatility,
    )
    
    # If position size ends up zero, skip
    if order_qty <= 0.0:
        return {}, [], 0, 0, 0, 0, 1
    
    recent_volumes.append(trade_size)
    
    # ==============================
    # 5) Fees
    # ==============================
    entry_fee = entry_price * order_qty * maker_fee

    # ==============================
    # 6) Trade tracking
    # ==============================
    position_remaining = order_qty
    average_entry_price = entry_price

    # For trailing stop: store best price (for long) or worst (for short)
    best_price_seen = entry_price if direction > 0 else -entry_price
    trailing_stop_activated = False

    # Results tracking
    trade_results = []
    exit_found = False
    slippage_exit = 0.0

    # ==============================
    # 7) Main exit loop
    # ==============================
    for idx in range(aligned_index + 1, n_rows):
        # Get base exit quote for this row
        if exit_price_type == 'AskPrice_1':
            base_exit_price = AskPrice1_array[idx] if idx < len(AskPrice1_array) else None
        else:
            base_exit_price = BidPrice1_array[idx] if idx < len(BidPrice1_array) else None
        
        if base_exit_price is None or np.isnan(base_exit_price):
            # No quote for this row, skip
            continue

        # Simulate exit slippage if needed
        exit_price = base_exit_price
        if simulate_slippage:
            try:
                lob_exit_row = lob_data.iloc[idx]
                bid_levels_exit, ask_levels_exit = extract_book_levels(lob_exit_row, depth=book_depth)

                if direction > 0:
                    exit_levels_sim = bid_levels_exit
                else:
                    exit_levels_sim = ask_levels_exit

                if exit_levels_sim:
                    exit_vwap, exit_filled_qty = simulate_fill_price(
                        position_remaining,
                        exit_levels_sim,
                        -direction,  # opposite direction to exit
                        latency_ms=latency,
                        concurrency_factor=concurrency_factor
                    )
                    if exit_vwap is not None and exit_filled_qty > 0:
                        exit_price = exit_vwap
                        slippage_exit = abs(exit_price - base_exit_price) / (base_exit_price + 1e-9)
            except (IndexError, KeyError):
                pass
        
        # Evaluate trailing stop or profit target
        exit_triggered, exit_type, realized_profit, exit_price, best_price_seen, trailing_stop_activated, exit_fee = process_trade_exits(
            direction,
            exit_price,
            average_entry_price,
            position_remaining,
            entry_fee,
            idx,
            aligned_index,
            best_price_seen if direction > 0 else -best_price_seen,
            trailing_stop_activated,
            params
        )

        if exit_triggered:
            # Build final trade result
            trade_result = create_trade_result(
                exit_type,
                realized_profit,
                average_entry_price,
                exit_price,
                slippage_entry,
                slippage_exit,
                position_remaining,
                aligned_index,
                idx,
                latency,
                trade_timestamp,
                direction,
                current_imbalance,
                entry_fee,
                exit_fee,
                detailed_logging
            )
            trade_results.append(trade_result)
            # Update equity curve
            equity_curve.append(equity_curve[-1] + realized_profit)
            exit_found = True

            # Score success vs fail
            if realized_profit > 0:
                return {}, [trade_result], realized_profit, realized_profit, 1, 0, 0
            else:
                return {}, [trade_result], realized_profit, 0, 0, 1, 0

    # ==============================
    # 8) If no exit triggered, mark to market at last row
    # ==============================
    if not exit_found:
        final_idx = n_rows - 1  # last row index
        # Attempt to retrieve a final quote
        if exit_price_type == 'AskPrice_1':
            final_exit_price = AskPrice1_array[final_idx] if final_idx < len(AskPrice1_array) else np.nan
        else:
            final_exit_price = BidPrice1_array[final_idx] if final_idx < len(BidPrice1_array) else np.nan

        # If we can't find a quote at the final row, treat it as skip
        if np.isnan(final_exit_price):
            # The strategy can't close if there's no final quote, so skip
            return {}, [], 0, 0, 0, 0, 1

        # Mark the position to market using the final row's quote
        price_change = (final_exit_price - average_entry_price) * direction
        exit_fee = final_exit_price * position_remaining * params.get('taker_fee', 0.0003)
        realized_profit = price_change * position_remaining - entry_fee - exit_fee

        trade_result = create_trade_result(
            'end_of_data',
            realized_profit,
            average_entry_price,
            final_exit_price,
            slippage_entry,
            slippage_exit,
            position_remaining,
            aligned_index,
            final_idx,
            latency,
            trade_timestamp,
            direction,
            current_imbalance,
            entry_fee,
            exit_fee,
            detailed_logging
        )
        trade_results.append(trade_result)
        equity_curve.append(equity_curve[-1] + realized_profit)

        # Score success vs fail
        if realized_profit > 0:
            return {}, [trade_result], realized_profit, realized_profit, 1, 0, 0
        else:
            return {}, [trade_result], realized_profit, 0, 0, 1, 0



# ===== 5. FULL BACKTEST FUNCTION =====

def backtest(
    sig_trades_aligned: pd.DataFrame,
    lob_data: pd.DataFrame,
    taker_fee: float = 0.0003,
    maker_fee: float = 0.0001,
    fee_multiple: float = 2.0,
    capital: float = 10_000.0,
    min_size: float = 0.5,
    max_size: float = 3.0,
    imb_threshold_long: float = -0.6,
    imb_threshold_short: float = 0.6,
    simulate_slippage: bool = False,
    mean_latency_ms: float = 50.0,
    std_dev_latency_ms: float = 20.0,
    book_depth: int = 1,
    slippage_factor: float = 1.0,
    detailed_logging: bool = True,
    recent_volume_window: int = 50,
    use_microprice: bool = True,
    use_trailing_stop: bool = True,
    trailing_activation_bps: float = 3.0,
    trailing_distance_bps: float = 1.0,
    max_risk_pct: float = 0.002
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Full-timeline backtest that steps through EVERY row of lob_data in chronological order:
      - Opens or scales up a position on each new signal if same direction.
      - If signal is opposite direction, closes the old position first, then opens new.
      - Updates the open position each row, applying trailing stop / profit target logic.
      - Closes at dataset end if still open, marking PnL at that final price.
    """

    lob_data = lob_data.sort_index()
    BidPrice1_array = lob_data['BidPrice_1'].values if 'BidPrice_1' in lob_data.columns else np.array([])
    AskPrice1_array = lob_data['AskPrice_1'].values if 'AskPrice_1' in lob_data.columns else np.array([])
    Imbalance_array = lob_data['Imbalance'].values if 'Imbalance' in lob_data.columns else np.array([])
    MidPrice_array = lob_data['MidPrice'].values if 'MidPrice' in lob_data.columns else np.array([])
    Microprice_array = lob_data['Microprice'].values if 'Microprice' in lob_data.columns else np.array([])
    n_rows = len(lob_data)
    if n_rows == 0:
        return {}, []

    sig_trades_aligned = sig_trades_aligned.sort_index()
    signals_list = list(sig_trades_aligned.itertuples())
    sig_i = 0
    total_signals = len(signals_list)

    trade_results = []
    equity_curve = [0.0]
    # 'open_pos' can now hold a scaled position: 
    #   direction, entry_price (avg), order_qty, entry_fee, etc.
    open_pos = None

    recent_volumes = deque(maxlen=recent_volume_window)
    total_profit = 0.0
    successful_trades = 0
    failed_trades = 0
    skipped_trades = 0

    params = {
        'taker_fee': taker_fee,
        'maker_fee': maker_fee,
        'fee_multiple': fee_multiple,
        'capital': capital,
        'min_size': min_size,
        'max_size': max_size,
        'imb_threshold_long': imb_threshold_long,
        'imb_threshold_short': imb_threshold_short,
        'simulate_slippage': simulate_slippage,
        'mean_latency_ms': mean_latency_ms,
        'std_dev_latency_ms': std_dev_latency_ms,
        'slippage_factor': slippage_factor,
        'detailed_logging': detailed_logging,
        'use_microprice': use_microprice,
        'use_trailing_stop': use_trailing_stop,
        'trailing_activation_bps': trailing_activation_bps,
        'trailing_distance_bps': trailing_distance_bps,
        'max_risk_pct': max_risk_pct
    }

    def close_position(
        row_i: int,
        exit_price: float,
        current_imbalance: float,
        label: str
    ) -> float:
        """
        Closes the existing open_pos immediately and records a trade.
        Returns the realized profit for tracking success/fail.
        """
        nonlocal open_pos, trade_results, equity_curve, successful_trades, failed_trades

        direction = open_pos['direction']
        qty = open_pos['order_qty']
        entry_fee = open_pos['entry_fee']
        price_change = (exit_price - open_pos['entry_price']) * direction
        exit_fee = exit_price * qty * taker_fee
        realized_profit = price_change * qty - entry_fee - exit_fee

        trade_result = create_trade_result(
            label,
            realized_profit,
            open_pos['entry_price'],
            exit_price,
            open_pos.get('slippage_entry', 0.0),
            0.0,  # exit slippage not fully modeled here
            qty,
            open_pos['row_opened'],
            row_i,
            0.0,
            open_pos['time_opened'],
            direction,
            current_imbalance,
            entry_fee,
            exit_fee,
            detailed_logging
        )
        trade_results.append(trade_result)
        equity_curve.append(equity_curve[-1] + realized_profit)
        if realized_profit > 0:
            successful_trades += 1
        else:
            failed_trades += 1

        # Flatten out
        open_pos = None
        return realized_profit

    for row_i in range(n_rows):
        current_time = lob_data.index[row_i]

        # 1) Check if new signals arrive up to this time
        while sig_i < total_signals:
            sig = signals_list[sig_i]
            signal_time = sig.Index
            if signal_time <= current_time:
                direction, base_entry_price, exit_price_type, current_imbalance, signal_strength = \
                    determine_trade_direction(
                        sig,
                        row_i,
                        Imbalance_array,
                        BidPrice1_array,
                        AskPrice1_array,
                        MidPrice_array,
                        Microprice_array,
                        params
                    )
                if direction is None:
                    skipped_trades += 1
                    sig_i += 1
                    continue

                # Compute entry price with slippage
                latency = sample_latency(mean_latency_ms, std_dev_latency_ms) if simulate_slippage else 0.0
                entry_price, slippage_entry = calculate_slippage_and_entry_price(
                    row_i, direction, base_entry_price, sig.qty, 
                    lob_data, book_depth, latency, slippage_factor, 0.0
                )
                if entry_price is None:
                    skipped_trades += 1
                    sig_i += 1
                    continue

                if len(recent_volumes) > 10:
                    recent_volatility = np.std(recent_volumes) + 1.0
                else:
                    recent_volatility = 1.0

                new_order_qty = conservative_position_sizing(
                    trade_size=sig.qty,
                    capital=capital,
                    entry_price=entry_price,
                    signal_strength=signal_strength,
                    max_risk_pct=max_risk_pct,
                    min_qty=min_size,
                    max_qty=max_size,
                    recent_volatility=recent_volatility
                )
                if new_order_qty <= 0.0:
                    skipped_trades += 1
                    sig_i += 1
                    continue

                # =========== KEY CHANGE: if open_pos is not None, 
                # we either scale or close & reverse ================
                if open_pos is None:
                    # Open new position from scratch
                    open_pos = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'order_qty': new_order_qty,
                        'entry_fee': entry_price * new_order_qty * maker_fee,
                        'row_opened': row_i,
                        'time_opened': current_time,
                        'best_price_seen': entry_price if direction>0 else -entry_price,
                        'trailing_stop_activated': False,
                        'slippage_entry': slippage_entry
                    }
                else:
                    # We have an existing position
                    current_dir = open_pos['direction']
                    if current_dir == direction:
                        # Same direction => scale up
                        old_qty = open_pos['order_qty']
                        old_entry_price = open_pos['entry_price']
                        old_fee = open_pos['entry_fee']
                        total_qty = old_qty + new_order_qty

                        # Weighted-average entry price
                        new_avg_price = (
                            old_qty * old_entry_price + new_order_qty * entry_price
                        ) / total_qty

                        # Additional fee for the new lot
                        add_fee = entry_price * new_order_qty * maker_fee

                        open_pos['direction'] = direction
                        open_pos['entry_price'] = new_avg_price
                        open_pos['order_qty'] = total_qty
                        open_pos['entry_fee'] = old_fee + add_fee
                        # best_price_seen might need adjusting if direction>0 
                        # (the best was the highest for a long) or direction<0
                        if direction>0:
                            open_pos['best_price_seen'] = max(
                                open_pos['best_price_seen'], new_avg_price
                            )
                        else:
                            open_pos['best_price_seen'] = min(
                                open_pos['best_price_seen'], -new_avg_price
                            )
                        # trailing_stop_activated remains as is
                    else:
                        # Opposite direction => close old pos, then open new
                        # We'll do it at the current row's best quotes
                        # to "simulate" immediate close
                        if current_dir>0:
                            # close at AskPrice_1
                            close_px = AskPrice1_array[row_i] if row_i<len(AskPrice1_array) else entry_price
                        else:
                            # close at BidPrice_1
                            close_px = BidPrice1_array[row_i] if row_i<len(BidPrice1_array) else entry_price
                        if not np.isnan(close_px):
                            close_position(row_i, close_px, current_imbalance, 'reverse_close')

                        # Now open the new position
                        open_pos = {
                            'direction': direction,
                            'entry_price': entry_price,
                            'order_qty': new_order_qty,
                            'entry_fee': entry_price * new_order_qty * maker_fee,
                            'row_opened': row_i,
                            'time_opened': current_time,
                            'best_price_seen': entry_price if direction>0 else -entry_price,
                            'trailing_stop_activated': False,
                            'slippage_entry': slippage_entry
                        }

                sig_i += 1
            else:
                break
        # End while signals

        # 2) If we have an open position, check trailing stop or profit target
        if open_pos is not None:
            direction = open_pos['direction']
            if direction>0 and row_i<len(AskPrice1_array):
                base_exit_price = AskPrice1_array[row_i]
            elif direction<0 and row_i<len(BidPrice1_array):
                base_exit_price = BidPrice1_array[row_i]
            else:
                base_exit_price = np.nan

            if not np.isnan(base_exit_price):
                exit_price = base_exit_price
                # If you want exit slippage, do it here
                price_change = (exit_price - open_pos['entry_price']) * direction
                exit_fee = exit_price * open_pos['order_qty'] * taker_fee
                net_pnl = price_change * open_pos['order_qty'] - open_pos['entry_fee'] - exit_fee

                updated_best_price = update_best_price(
                    exit_price, open_pos['best_price_seen'], direction
                )
                updated_trailing_activated = open_pos['trailing_stop_activated']

                # Check profit target
                round_trip_fees = open_pos['entry_fee'] + exit_fee
                exit_triggered = False
                exit_type = ''
                realized_profit = 0.0
                if net_pnl >= fee_multiple * round_trip_fees:
                    exit_triggered = True
                    exit_type = 'profit_target'
                    realized_profit = net_pnl
                else:
                    # trailing stop
                    if use_trailing_stop:
                        trailing_activation = trailing_activation_bps / 10000.0
                        trailing_distance = trailing_distance_bps / 10000.0
                        price_change_pct = price_change / open_pos['entry_price']
                        if price_change_pct >= trailing_activation and not updated_trailing_activated:
                            updated_trailing_activated = True
                        if updated_trailing_activated:
                            if check_trailing_stop(exit_price, updated_best_price, direction, trailing_distance):
                                exit_triggered = True
                                exit_type = 'trailing_stop'
                                realized_profit = net_pnl

                open_pos['best_price_seen'] = updated_best_price
                open_pos['trailing_stop_activated'] = updated_trailing_activated
                if exit_triggered:
                    # record
                    trade_result = create_trade_result(
                        exit_type,
                        realized_profit,
                        open_pos['entry_price'],
                        exit_price,
                        open_pos.get('slippage_entry', 0.0),
                        0.0,  # exit slippage
                        open_pos['order_qty'],
                        open_pos['row_opened'],
                        row_i,
                        0.0,
                        open_pos['time_opened'],
                        direction,
                        Imbalance_array[row_i] if row_i < len(Imbalance_array) else 0.0,
                        open_pos['entry_fee'],
                        exit_fee,
                        detailed_logging
                    )
                    trade_results.append(trade_result)
                    equity_curve.append(equity_curve[-1] + realized_profit)
                    if realized_profit>0:
                        successful_trades += 1
                    else:
                        failed_trades += 1
                    open_pos = None
        # end if open_pos

        # potentially update recent_volumes, etc. if you want

    # 3) If still open at the end, close at last row
    if open_pos is not None:
        last_row = n_rows - 1
        if open_pos['direction']>0:
            final_exit_price = AskPrice1_array[last_row]
        else:
            final_exit_price = BidPrice1_array[last_row]
        if not np.isnan(final_exit_price):
            close_position(last_row, final_exit_price, 0.0, 'end_of_data')
        else:
            skipped_trades += 1
        open_pos = None

    total_profit = sum(tr['profit'] for tr in trade_results)
    summary = compile_backtest_results(
        total_profit,
        successful_trades,
        failed_trades,
        skipped_trades,
        0.0, # total_slippage_entry
        0.0, # total_slippage_exit
        0,
        trade_results,
        capital
    )
    return summary, trade_results



# ===== 6. SENSITIVITY ANALYSIS =====

def sensitivity_analysis(
    significant_trades_aligned: pd.DataFrame,
    lob_data: pd.DataFrame,
    base_params: Dict[str, Any],
    param_ranges: Dict[str, List[Any]],
    num_trades: int = 1000
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on the strategy.
    """
    if len(significant_trades_aligned) > num_trades:
        sampled_trades = significant_trades_aligned.sample(num_trades)
    else:
        sampled_trades = significant_trades_aligned.copy()

    results = []
    param_names = list(param_ranges.keys())

    for param_name in param_names:
        param_values = param_ranges[param_name]

        for param_value in param_values:
            test_params = base_params.copy()
            test_params[param_name] = param_value

            summary, _ = backtest(sampled_trades, lob_data, **test_params)
            result = {
                'parameter': param_name,
                'value': param_value,
                **summary
            }
            results.append(result)

    return pd.DataFrame(results)


def apply_multi_test_correction(
    sensitivity_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "holm"
) -> pd.DataFrame:
    """
    Applies multiple-hypothesis p-value corrections to sensitivity results.

    For demonstration, we assume a 'pvalue' column is present.
    """
    if 'pvalue' not in sensitivity_df.columns:
        logging.warning("No 'pvalue' column found in sensitivity_df. Can't apply multi-test correction.")
        return sensitivity_df

    df = sensitivity_df.copy().reset_index(drop=True)
    pvals = df['pvalue'].values
    m = len(pvals)

    # Simple Holm–Bonferroni
    if method.lower() == "holm":
        sorted_idx = np.argsort(pvals)
        adjusted = np.empty(m, dtype=float)
        current_max = 0.0
        for i, idx in enumerate(sorted_idx):
            p_adj = (m - i) * pvals[idx]
            current_max = max(current_max, p_adj)
            adjusted[idx] = min(current_max, 1.0)
        df['pvalue_corrected'] = adjusted
        df['significant'] = df['pvalue_corrected'] < alpha

    elif method.lower() == "bonferroni":
        df['pvalue_corrected'] = np.minimum(pvals * m, 1.0)
        df['significant'] = df['pvalue_corrected'] < alpha
    else:
        logging.warning(f"Method '{method}' not recognized. No correction applied.")
        df['pvalue_corrected'] = pvals
        df['significant'] = pvals < alpha

    return df


# ===== 7. WALK-FORWARD TEST =====

def walk_forward_test(
    significant_trades_aligned: pd.DataFrame,
    lob_data: pd.DataFrame,
    base_params: Dict[str, Any],
    param_name: Optional[str] = None,
    param_values: Optional[List[Any]] = None,
    train_size: float = 0.7,
    step_size: float = 0.1
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Perform walk-forward testing to validate strategy robustness, optionally re-optimizing a single parameter.

    If `param_name` and `param_values` are provided, the function attempts each value in the TRAIN window,
    picks the best (by total profit), then applies that value to the TEST window.
    """
    significant_trades_aligned = significant_trades_aligned.sort_values('aligned_index')
    n_trades = len(significant_trades_aligned)
    if n_trades < 30:
        logging.info("Not enough trades to do a meaningful walk-forward.")
        return [], pd.DataFrame()

    train_trades = int(n_trades * train_size)
    step_trades = max(1, int(n_trades * step_size))

    period_results = []
    all_trade_results = []

    i = 0
    period_idx = 1
    while i + train_trades < n_trades:
        train_end = i + train_trades
        test_end = min(train_end + step_trades, n_trades)

        train_df = significant_trades_aligned.iloc[i:train_end]
        test_df = significant_trades_aligned.iloc[train_end:test_end]

        if train_df.empty or test_df.empty:
            break

        logging.info(f"Walk-forward period {period_idx}: train_size={len(train_df)}, test_size={len(test_df)}")

        # 1) If we have param_name/param_values, do an optimization on train slice
        best_param_value = None
        best_profit = -1e10

        if param_name and param_values:
            for val in param_values:
                test_params = base_params.copy()
                test_params[param_name] = val
                train_summary, _ = backtest(train_df, lob_data, **test_params)
                if train_summary['Total Profit'] > best_profit:
                    best_profit = train_summary['Total Profit']
                    best_param_value = val
        else:
            best_param_value = base_params.get(param_name, None)

        # 2) Now run the TEST slice with the best param
        final_params = base_params.copy()
        if param_name and best_param_value is not None:
            final_params[param_name] = best_param_value

        test_summary, test_trades_list = backtest(test_df, lob_data, **final_params)

        # Collect period summary
        period_info = {
            'period': period_idx,
            'train_start': i,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'Train Best Param': best_param_value,
            'Train Best Profit': best_profit,
            **test_summary
        }
        period_results.append(period_info)

        if test_trades_list:
            test_results_df = pd.DataFrame(test_trades_list)
            test_results_df['period'] = period_idx
            all_trade_results.append(test_results_df)

        i = test_end
        period_idx += 1

        if test_end >= n_trades:
            break

    combined_results = pd.concat(all_trade_results, ignore_index=True) if all_trade_results else pd.DataFrame()
    return period_results, combined_results


# ===== 8. VISUALIZATION =====

def plot_results(
    df_results: pd.DataFrame,
    title: str = "Backtest Results"
) -> None:
    """
    Plots backtest results.
    """
    if df_results.empty:
        logging.info("No results to plot.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    fig.suptitle(title, fontsize=14)

    # Plot 1: Cumulative Profit
    axs[0].plot(df_results.index, df_results['cumulative_profit'], color='steelblue')
    axs[0].set_title('Cumulative Profit')
    axs[0].set_xlabel('Trade #')
    axs[0].set_ylabel('Profit')
    axs[0].grid(alpha=0.3)

    # Plot 2: Profit Distribution
    axs[1].hist(df_results['profit'], bins=30, alpha=0.7, color='pink', edgecolor='black')
    axs[1].axvline(x=0, color='red', linestyle='--')
    axs[1].set_title('Profit Distribution')
    axs[1].set_xlabel('Profit')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(alpha=0.3)

    # Plot 3: Holding Period Distribution
    if 'holding_period' in df_results.columns:
        axs[2].hist(df_results['holding_period'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axs[2].set_title('Holding Period Distribution')
        axs[2].set_xlabel('Holding Period (snapshots)')
        axs[2].set_ylabel('Frequency')
        axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_sensitivity_analysis(
    sensitivity_results: pd.DataFrame,
    metric: str = 'Total Profit'
) -> None:
    """
    Plots sensitivity analysis results.
    """
    if sensitivity_results.empty:
        logging.info("No sensitivity results to plot.")
        return

    parameters = sensitivity_results['parameter'].unique()
    fig, axs = plt.subplots(len(parameters), 1, figsize=(8, 3 * len(parameters)))
    if len(parameters) == 1:
        axs = [axs]

    for i, param in enumerate(parameters):
        param_results = sensitivity_results[sensitivity_results['parameter'] == param]
        param_results = param_results.sort_values('value')

        axs[i].plot(param_results['value'], param_results[metric], 'o-', color='steelblue')
        axs[i].set_title(f'Effect of {param}')
        axs[i].set_xlabel(param)
        axs[i].set_ylabel(metric)
        axs[i].grid(alpha=0.3)

        for x, y in zip(param_results['value'], param_results[metric]):
            axs[i].annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                            xytext=(0, 5), ha='center', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_bootstrap_analysis(
    returns: np.ndarray,
    bootstrap_mean: float,
    conf_interval: Tuple[float, float],
    title: str = "Bootstrap Analysis"
) -> None:
    """
    Plots bootstrap analysis results (using a separate internal i.i.d. sample).
    """
    if len(returns) == 0:
        logging.info("No returns to plot.")
        return

    plt.figure(figsize=(7, 4))

    # We create a quick naive distribution for the visuals
    bootstrap_samples = []
    for _ in range(1000):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_samples.append(np.mean(sample))

    plt.hist(bootstrap_samples, bins=40, alpha=0.7, color='pink', edgecolor='black')

    plt.axvline(x=bootstrap_mean, color='red', linestyle='-', label=f'Mean: {bootstrap_mean:.4f}')
    plt.axvline(x=conf_interval[0], color='black', linestyle='--',
                label=f'95% CI: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]')
    plt.axvline(x=conf_interval[1], color='black', linestyle='--')
    plt.axvline(x=0, color='green', linestyle='-', label='Zero')

    plt.title(title)
    plt.xlabel('Mean Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_walk_forward_results(
    period_results: List[Dict[str, Any]],
    metric: str = 'Total Profit'
) -> None:
    """
    Plots walk-forward test results.
    """
    if not period_results:
        logging.info("No walk-forward results to plot.")
        return

    results_df = pd.DataFrame(period_results)
    plt.figure(figsize=(8, 4))

    plt.plot(results_df['period'], results_df[metric], 'o-', color='steelblue')
    plt.title(f'Walk-Forward Test: {metric}')
    plt.xlabel('Period')
    plt.ylabel(metric)
    plt.grid(alpha=0.3)

    for i, v in enumerate(results_df[metric]):
        plt.annotate(f'{v:.2f}', (results_df['period'].iloc[i], v),
                     textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    plt.show()


def compare_results(
    results_with_slippage: Dict[str, Any],
    results_without_slippage: Dict[str, Any]
) -> None:
    """
    Compares backtest results with and without slippage.
    """
    metrics = ['Total Profit', 'Win Rate', 'Successful Trades', 'Failed Trades']
    plt.figure(figsize=(9, 5))

    x = np.arange(len(metrics))
    width = 0.35

    with_slippage_values = [results_with_slippage.get(m, 0) for m in metrics]
    without_slippage_values = [results_without_slippage.get(m, 0) for m in metrics]

    plt.bar(x - width/2, with_slippage_values, width, label='With Slippage', color='steelblue')
    plt.bar(x + width/2, without_slippage_values, width, label='Without Slippage', color='orange')

    plt.title('Comparison: With vs. Without Slippage')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')

    for i, v in enumerate(with_slippage_values):
        if isinstance(v, float) and abs(v) > 1000:
            txt = f'{v:.0f}'
        elif isinstance(v, float):
            txt = f'{v:.4f}'
        else:
            txt = str(v)
        plt.text(i - width/2, v, txt, ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(without_slippage_values):
        if isinstance(v, float) and abs(v) > 1000:
            txt = f'{v:.0f}'
        elif isinstance(v, float):
            txt = f'{v:.4f}'
        else:
            txt = str(v)
        plt.text(i + width/2, v, txt, ha='center', va='bottom', fontsize=9)

    plt.show()


def sensitivity_analysis_with_pvalues(
    significant_trades_aligned: pd.DataFrame,
    lob_data: pd.DataFrame,
    base_params: Dict[str, Any],
    param_ranges: Dict[str, List[Any]],
    num_trades: int = 1000
) -> pd.DataFrame:
    """
    Perform sensitivity analysis with p-value calculation.
    """
    if len(significant_trades_aligned) > num_trades:
        sampled_trades = significant_trades_aligned.sample(num_trades)
    else:
        sampled_trades = significant_trades_aligned.copy()

    results = []
    param_names = list(param_ranges.keys())

    for param_name in param_names:
        param_values = param_ranges[param_name]

        for param_value in param_values:
            test_params = base_params.copy()
            test_params[param_name] = param_value

            summary, trade_list = backtest(sampled_trades, lob_data, **test_params)

            # Example: compute returns array
            if trade_list:
                df_trades = pd.DataFrame(trade_list)
                returns = df_trades['profit'] / base_params['capital']
                # Basic one-sample t-test vs 0 returns
                _, pval = ttest_1samp(returns, popmean=0.0)
            else:
                pval = 1.0

            result = {
                'parameter': param_name,
                'value': param_value,
                'pvalue': pval,
                **summary
            }
            results.append(result)

    return pd.DataFrame(results)