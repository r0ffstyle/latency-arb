"""
Latency Arbitrage - Data Loading Module

This module focuses exclusively on data loading, processing, and preparation functions.
It handles loading market data from files, processing order book data, and aligning trades.
"""

import os
import logging
import pandas as pd
import numpy as np
from numba import jit, cuda
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import orjson
import gc

# ===== 1. DATA LOADING FUNCTIONS =====

def load_binance_trades_monthly(
    symbol: str,
    year: int,
    start_month: int,
    end_month: int,
    base_path: str = "C:/Users/trgrd/OneDrive/Trading/Projects/data_download/Binance_data",
    file_extension: str = "csv",  # Accepts either 'csv' or 'parquet'
    chunksize: int = 1_000_000
) -> pd.DataFrame:
    """
    Loads Binance trades data for a given period.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'SOL')
    year : int
        Year to load data for
    start_month : int
        First month to load (1-12)
    end_month : int
        Last month to load (1-12)
    base_path : str
        Base directory where data is stored
    file_extension : str
        File extension to use ('csv' or 'parquet')
    chunksize : int
        Number of rows to read at once (used only for CSV files)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Binance trades data
    """
    
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/binance_{symbol}_{year}_{start_month}_{end_month}.pkl"

    # Return cached data if available
    if os.path.exists(cache_file):
        logging.info(f"Loading Binance trades from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    # Construct file paths based on the file_extension
    file_paths = [
        f"{base_path}/{symbol}/{symbol}USDT-trades-{year}-{month:02}.{file_extension.lower()}"
        for month in range(start_month, end_month + 1)
    ]

    binance_trades_list = []
    for i, path in enumerate(file_paths, start=1):
        logging.info(f"Loading Binance monthly file {i}/{len(file_paths)}: {path}")
        try:
            if file_extension.lower() == "csv":
                # Process CSV files in chunks for large files
                chunks = []
                for chunk in pd.read_csv(
                    path,
                    sep=",",
                    header=None,
                    names=["trade_id", "price", "quantity", "quote_quantity", "timestamp", "is_buyer_maker", "ignore"],
                    chunksize=chunksize
                ):
                    chunks.append(chunk)
                if chunks:
                    month_df = pd.concat(chunks, ignore_index=True)
                else:
                    continue
            elif file_extension.lower() == "parquet":
                # Directly load parquet file as chunking isn't supported
                month_df = pd.read_parquet(path)
            else:
                raise ValueError("Unsupported file extension. Please use 'csv' or 'parquet'.")

            # Convert timestamp to datetime if not already in datetime format
            # For CSV, the timestamp is assumed to be in milliseconds.
            # Adjust this if your parquet files store timestamps differently.
            if month_df['timestamp'].dtype != 'datetime64[ns]':
                month_df['timestamp'] = pd.to_datetime(month_df['timestamp'], unit='ms', errors='coerce')

            month_df.set_index('timestamp', inplace=True)
            month_df.sort_index(inplace=True)
            month_df = month_df[~month_df.index.duplicated(keep='last')]
            binance_trades_list.append(month_df)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")

    if binance_trades_list:
        binance_trades = pd.concat(binance_trades_list, ignore_index=False)
        binance_trades.sort_index(inplace=True)
        logging.info(f"Final Binance trades shape: {binance_trades.shape}")
        binance_trades.to_pickle(cache_file)
        logging.info(f"Cached Binance trades to: {cache_file}")
    else:
        binance_trades = pd.DataFrame()
        logging.info("No Binance trade data loaded at all.")

    return binance_trades

def load_processed_lob_data(
    symbol: str, 
    year: int, 
    start_month: int, 
    end_month: int, 
    base_path: str = "D:/data/processed_lob_data",
    memory_efficient: bool = True,
    columns_subset: list = None  # Optional: load only specific columns
) -> pd.DataFrame:
    """
    Memory-optimized version that loads processed LOB data from Parquet files.
    Fixed to properly handle memory-efficient mode.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'SOL')
    year : int
        Year of the data (e.g., 2024)
    start_month : int
        Starting month (1-12)
    end_month : int
        Ending month (1-12)
    base_path : str, optional
        Base directory where processed LOB data is stored
    memory_efficient : bool
        If True, uses memory optimization techniques
    columns_subset : list
        List of specific columns to load (None loads all columns)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the processed LOB data for the specified range.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache file name
    cols_hash = hash(str(columns_subset)) if columns_subset else "all"
    cache_file = f"{cache_dir}/lob_{symbol}_{year}_{start_month:02}_{end_month:02}_{cols_hash}.parquet.gz"
    
    # Check if cache file exists
    if os.path.exists(cache_file):
        logging.info(f"Loading LOB data from cache: {cache_file}")
        
        # For memory efficiency, consider using filters
        if memory_efficient and columns_subset:
            return pd.read_parquet(cache_file, columns=columns_subset)
        else:
            return pd.read_parquet(cache_file)
    
    # Build the output directory path
    output_dir = os.path.join(base_path, symbol)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate expected file names based on the year and month range
    expected_files = [
        os.path.join(output_dir, f"{symbol}_LOB_{year}{month:02}.parquet.gz")
        for month in range(start_month, end_month + 1)
    ]
    
    data_list = []
    total_files = len(expected_files)
    
    for i, f in enumerate(expected_files):
        logging.info(f"Loading monthly LOB file ({i+1}/{total_files}): {f}")
        if os.path.exists(f):
            try:
                if memory_efficient:
                    # If memory optimized, read only specific columns
                    chunk = pd.read_parquet(f, columns=columns_subset)
                    
                    # Convert float64 to float32 to reduce memory
                    for col in chunk.select_dtypes(include=['float64']).columns:
                        chunk[col] = chunk[col].astype('float32')
                else:
                    chunk = pd.read_parquet(f)
                    
                data_list.append(chunk)
                
                # Clear chunk to free memory
                del chunk
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error loading {f}: {e}")
        else:
            logging.warning(f"File not found: {f}")
    
    if data_list:
        # FIXED: If memory efficient mode, use a safer approach
        if memory_efficient and len(data_list) > 1:
            # Start with the first dataframe
            lob_data = data_list[0]
            
            # Use a safer approach that doesn't modify the list during iteration
            for i in range(1, len(data_list)):
                # Get the next dataframe
                next_df = data_list[i]
                
                # Concatenate with the current result
                lob_data = pd.concat([lob_data, next_df], ignore_index=False)
                
                # Free memory from the individual dataframe
                next_df = None
                gc.collect()
                
            # Clear the entire list to free memory
            data_list = None
        else:
            lob_data = pd.concat(data_list, ignore_index=False)
            
        lob_data.sort_index(inplace=True)
        logging.info(f"Final LOB shape: {lob_data.shape}")
        
        # Cache result (but not more than 2 months by default to prevent huge files)
        if (end_month - start_month + 1) <= 2 or not memory_efficient:
            lob_data.to_parquet(cache_file, compression='gzip')
            logging.info(f"Cached LOB data to: {cache_file}")
    else:
        lob_data = pd.DataFrame()
        logging.info("No LOB data loaded from monthly files.")
    
    return lob_data

# Function to select only essential columns for backtest to reduce memory usage
def load_minimal_lob_data(symbol, year, start_month, end_month, base_path="D:/data/processed_lob_data"):
    """Loads only essential columns for backtest to save memory"""
    
    # Determine the minimal set of columns needed for the backtest
    essential_columns = [
        'BidPrice_1', 'BidSize_1', 'AskPrice_1', 'AskSize_1', 
        'MidPrice', 'Imbalance', 'Microprice'
    ]
    
    return load_processed_lob_data(
        symbol, 
        year, 
        start_month, 
        end_month, 
        base_path=base_path,
        memory_efficient=True,
        columns_subset=essential_columns
    )

def process_lob_data(
    symbol: str,
    year: int,
    start_month: int,
    end_month: int,
    lob_base_path: str = "D:/data",
    output_base_path: str = "D:/data/processed_lob_data",
    batch_size: int = 5,     # Reduced default batch size drastically
    use_gpu: bool = False,   # Default to CPU to save memory during startup
    n_workers: int = 1       # Start with just one worker
) -> None:
    """
    Processes raw LOB data and saves as Parquet files with GPU acceleration.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'SOL')
    year : int
        Year to process data for
    start_month : int
        First month to process (1-12)
    end_month : int
        Last month to process (1-12)
    lob_base_path : str
        Base directory where raw LOB data is stored
    output_base_path : str
        Base directory where processed LOB data will be saved
    batch_size : int
        Number of files to process in a single batch
    use_gpu : bool
        Whether to use GPU acceleration
    n_workers : int
        Number of parallel workers
    """
    # Force garbage collection at start
    gc.collect()
    
    print(f"Starting LOB processing with low memory settings")
    print(f"Symbol: {symbol}, Year: {year}, Months: {start_month}-{end_month}")
    
    output_dir = f"{output_base_path}/{symbol}"
    os.makedirs(output_dir, exist_ok=True)

    # Memory-efficient check for existing files
    print("Checking for existing processed files...")
    months_to_process = []
    
    for month in range(start_month, end_month + 1):
        output_file = f"{symbol}_LOB_{year}{month:02}.parquet.gz"
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(output_path):
            months_to_process.append(month)
    
    if not months_to_process:
        print(f"✓ LOB data already processed for {symbol} from {year}-{start_month} to {year}-{end_month}")
        return
    
    print(f"Will process {len(months_to_process)} months: {months_to_process}")
    
    # Process ONE month at a time - much more memory efficient
    for month in months_to_process:
        try:
            output_path = os.path.join(output_dir, f"{symbol}_LOB_{year}{month:02}.parquet.gz")
            
            print(f"\n===== Processing month {year}-{month} =====")
            result = process_month(
                symbol, 
                year, 
                month,
                lob_base_path,
                output_path, 
                batch_size=batch_size, 
                use_gpu=use_gpu
            )
            
            # Force garbage collection between months
            gc.collect()
            
            if result:
                print(f"✓ Completed month {year}-{month}: {result}")
            else:
                print(f"⚠️ Failed to process month {year}-{month}")
        except Exception as e:
            print(f"❌ Error processing month {year}-{month}: {e}")
            # Continue with next month

def generate_month_file_paths(symbol, year, month, lob_base_path):
    """Generate file paths for a specific month with minimal memory usage"""
    month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    
    # Handle leap year
    is_leap_year = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    if is_leap_year and month == 2:
        days = 29
    else:
        days = month_days[month]
    
    month_str = f"{year}{month:02}"
    base_dir = f"{lob_base_path}/{symbol}/monthly_pickled_data/{month_str}"
    
    # Quick check if directory exists first
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return []
    
    # Check each file without building a large list in memory
    existing_files = []
    for day in range(1, days + 1):
        path = f"{base_dir}/{month_str}{day:02}.pkl"
        if os.path.exists(path) and os.path.getsize(path) > 0:  # Skip empty files
            existing_files.append(path)
            # Print status periodically but not for every file
            if len(existing_files) == 1 or len(existing_files) % 5 == 0:
                print(f"Found {len(existing_files)} files so far for {month_str}")
    
    print(f"Found {len(existing_files)}/{days} files for {month_str}")
    return existing_files

@cuda.jit
def calculate_metrics_cuda(bid_px, bid_sz, ask_px, ask_sz, mid_price, imbalance, microprice):
    """CUDA kernel for financial metrics calculation"""
    i = cuda.grid(1)
    if i < bid_px.size:
        # Calculate mid price
        if not np.isnan(bid_px[i]) and not np.isnan(ask_px[i]):
            mid_price[i] = (bid_px[i] + ask_px[i]) / 2
        else:
            mid_price[i] = np.nan
            
        # Calculate imbalance and microprice
        total_sz = bid_sz[i] + ask_sz[i]
        if not np.isnan(total_sz) and total_sz != 0:
            imbalance[i] = (ask_sz[i] - bid_sz[i]) / total_sz
            microprice[i] = (bid_sz[i] * ask_px[i] + ask_sz[i] * bid_px[i]) / total_sz
        else:
            imbalance[i] = np.nan
            microprice[i] = np.nan

@jit(nopython=True, parallel=True)
def calculate_metrics_numba(bid_px, bid_sz, ask_px, ask_sz):
    """Calculate financial metrics using Numba for CPU acceleration"""
    n = len(bid_px)
    mid_price = np.empty(n, dtype=np.float32)
    imbalance = np.empty(n, dtype=np.float32)
    microprice = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        if np.isnan(bid_px[i]) or np.isnan(ask_px[i]):
            mid_price[i] = np.nan
        else:
            mid_price[i] = (bid_px[i] + ask_px[i]) / 2
            
        total_sz = bid_sz[i] + ask_sz[i]
        if np.isnan(total_sz) or total_sz == 0:
            imbalance[i] = np.nan
            microprice[i] = np.nan
        else:
            imbalance[i] = (ask_sz[i] - bid_sz[i]) / total_sz
            microprice[i] = (bid_sz[i] * ask_px[i] + ask_sz[i] * bid_px[i]) / total_sz
            
    return mid_price, imbalance, microprice

def calculate_metrics_gpu(bid_px, bid_sz, ask_px, ask_sz):
    """Calculate financial metrics using direct CUDA kernels"""
    # Allocate memory on device
    d_bid_px = cuda.to_device(bid_px)
    d_bid_sz = cuda.to_device(bid_sz)
    d_ask_px = cuda.to_device(ask_px)
    d_ask_sz = cuda.to_device(ask_sz)
    
    # Output arrays
    d_mid_price = cuda.device_array_like(bid_px)
    d_imbalance = cuda.device_array_like(bid_px)
    d_microprice = cuda.device_array_like(bid_px)
    
    # Define grid and block sizes
    threads_per_block = 256
    blocks_per_grid = (bid_px.size + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    calculate_metrics_cuda[blocks_per_grid, threads_per_block](
        d_bid_px, d_bid_sz, d_ask_px, d_ask_sz, 
        d_mid_price, d_imbalance, d_microprice
    )
    
    # Copy results back to host
    mid_price = d_mid_price.copy_to_host()
    imbalance = d_imbalance.copy_to_host()
    microprice = d_microprice.copy_to_host()
    
    return mid_price, imbalance, microprice

def extract_json_batch(json_strs, use_gpu=True):
    """Extract data from JSON strings in a batch-optimized way"""
    n = len(json_strs)
    coins = [None] * n
    times = [None] * n
    levels = [[[], []]] * n
    
    # Use vectorized operations where possible
    for i, s in enumerate(json_strs):
        try:
            # Use orjson for faster parsing
            row_data = orjson.loads(s)['raw']['data']
            coins[i] = row_data['coin']
            times[i] = row_data['time']
            levels[i] = row_data.get('levels', [[], []])
        except:
            pass  # Keep default None values
    
    return coins, times, levels

def get_top_level_vectorized(data, field):
    """Vectorized extraction of top level data from LOB"""
    result = np.full(len(data), np.nan, dtype=np.float32)
    
    for i, x in enumerate(data):
        if x and len(x) > 0 and isinstance(x[0], dict) and field in x[0]:
            result[i] = x[0][field]
    
    return result

def process_pickle_file(path):
    """Process a single pickle file with optimized I/O"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return None
        
        if df.empty:
            return None
        
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

def load_and_process_batch(file_paths, use_gpu=False):
    """Load and process a batch of pickle files with minimal memory usage"""
    all_data = []
    print(f"Loading {len(file_paths)} files (memory-efficient mode)...")
    
    # Process ONE FILE AT A TIME to minimize memory usage
    for i, path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(path)}")
        
        try:
            # Load a single pickle file
            df = process_pickle_file(path)
            
            if df is None or df.empty:
                print(f"  - Empty or invalid file, skipping")
                continue
                
            # Extract JSON strings - process in small chunks
            row_count = len(df)
            chunk_size = min(1000, row_count)  # Process at most 1000 rows at a time
            
            processed_chunks = []
            
            for chunk_start in range(0, row_count, chunk_size):
                chunk_end = min(chunk_start + chunk_size, row_count)
                print(f"  - Processing rows {chunk_start} to {chunk_end} of {row_count}")
                
                # Get slice of data
                json_str = df.iloc[chunk_start:chunk_end, 0].astype(str).values
                
                # Process this chunk
                coins, times, levels = extract_json_batch(json_str, use_gpu=False)  # Always use CPU for extraction
                
                # Create normalized DataFrame
                df_norm = pd.DataFrame({
                    'coin': coins,
                    'time': pd.to_datetime(times, unit='ms', errors='coerce'),
                    'levels': levels
                }).dropna(subset=['time']).sort_values('time')
                
                if df_norm.empty:
                    continue
                
                # Extract bid/ask data efficiently
                bids = [lvls[0] if isinstance(lvls, list) and len(lvls) > 0 else [] for lvls in df_norm['levels']]
                asks = [lvls[1] if isinstance(lvls, list) and len(lvls) > 1 else [] for lvls in df_norm['levels']]
                
                # Get top level prices and sizes
                bid_px = get_top_level_vectorized(bids, 'px')
                bid_sz = get_top_level_vectorized(bids, 'sz')
                ask_px = get_top_level_vectorized(asks, 'px')
                ask_sz = get_top_level_vectorized(asks, 'sz')
                
                # Calculate metrics - prefer CPU for small chunks to avoid GPU overhead
                mid_price, imbalance, microprice = calculate_metrics_numba(bid_px, bid_sz, ask_px, ask_sz)
                
                # Create processed dataframe
                df_processed = pd.DataFrame({
                    'coin': df_norm['coin'].values,
                    'BidPrice_1': bid_px,
                    'BidSize_1': bid_sz,
                    'AskPrice_1': ask_px,
                    'AskSize_1': ask_sz,
                    'MidPrice': mid_price,
                    'Imbalance': imbalance,
                    'Microprice': microprice,
                    'levels': df_norm['levels'].values
                }, index=df_norm['time']).sort_index()
                
                processed_chunks.append(df_processed)
                
                # Clear temporary variables
                del df_norm, bids, asks, bid_px, bid_sz, ask_px, ask_sz, mid_price, imbalance, microprice
            
            # Combine chunks from this file
            if processed_chunks:
                file_df = pd.concat(processed_chunks)
                all_data.append(file_df)
                print(f"  ✓ Processed file: {len(file_df)} rows")
                
                # Clear memory
                del file_df
                del processed_chunks
            
            # Clear original dataframe
            del df
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
    
    # Combine results from all files in this batch
    if all_data:
        print(f"Combining results from {len(all_data)} files...")
        if len(all_data) == 1:
            result = all_data[0]
        else:
            result = pd.concat(all_data)
        print(f"✓ Processed batch: {len(result)} total rows")
        return result
    
    return None

def process_month(symbol, year, month, lob_base_path, output_path, batch_size=5, use_gpu=False):
    """Process all files for a given month in optimized batches with minimal memory usage,
    writing each batch incrementally to a Parquet file using PyArrow."""
    import gc
    gc.collect()
    
    # Get file paths for this month
    print(f"Getting file paths for {symbol} {year}-{month}...")
    file_paths = generate_month_file_paths(symbol, year, month, lob_base_path)
    
    if not file_paths:
        print(f"⚠️ No files found for {symbol} {year}-{month}")
        return None

    # Prepare the ParquetWriter (will be initialized with the schema from the first valid batch)
    writer = None
    total_batches = (len(file_paths) + batch_size - 1) // batch_size

    # Process files in small batches to manage memory
    for i in range(0, len(file_paths), batch_size):
        batch_number = i // batch_size + 1
        batch_paths = file_paths[i:i+batch_size]
        print(f"Processing batch {batch_number}/{total_batches} for {symbol} {year}-{month}")

        df_batch = load_and_process_batch(batch_paths, use_gpu)
        
        if df_batch is not None and not df_batch.empty:
            print(f"✓ Batch {batch_number} processed: {len(df_batch)} rows")
            
            # Convert the DataFrame to a PyArrow Table
            table = pa.Table.from_pandas(df_batch, preserve_index=True)
            
            # Initialize the writer if it hasn't been already
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='gzip')
            
            # Write this table (batch) incrementally
            writer.write_table(table)
            
            # Clear the batch DataFrame from memory
            del df_batch, table
        else:
            print(f"⚠️ Batch {batch_number} produced no data")
        
        gc.collect()

    # If a writer was created, close it now
    if writer is not None:
        writer.close()
        print(f"✓ Successfully wrote data for {symbol} {year}-{month} incrementally")
        return output_path
    else:
        print(f"⚠️ No valid data processed for {symbol} {year}-{month}")
        return None

# ===== 2. TRADE IDENTIFICATION AND ALIGNMENT =====

def identify_significant_trades(
    binance_trades: pd.DataFrame,
    quantile: float = 0.999,
    use_rolling_threshold: bool = False,
    rolling_window: str = '1h',
    rolling_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Identifies significant trades. Either uses a global quantile threshold (default)
    or a rolling threshold based on local volatility or local average + X*std.

    If use_rolling_threshold=True, we compute rolling mean and rolling std of quantity over the specified window,
    and consider trades "significant" if quantity > (mean + rolling_multiplier*std).

    Parameters:
    -----------
    binance_trades : pd.DataFrame
        Must have column 'quantity'
    quantile : float
        If not using rolling threshold, we pick trades above this quantile
    use_rolling_threshold : bool
        If True, we do a rolling-based threshold instead of global quantile
    rolling_window : str
        E.g., '1h' or '30min'. A pandas offset for rolling.
    rolling_multiplier : float
        If quantity > mean + multiplier*std in the window, we call it significant.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with significant trades
    """
    if binance_trades.empty:
        return pd.DataFrame()

    if not use_rolling_threshold:
        q_threshold = binance_trades['qty'].quantile(quantile)
        sig_trades = binance_trades[binance_trades['qty'] > q_threshold]
        logging.info(f"Identified {len(sig_trades)} significant trades above quantile {quantile}")
        return sig_trades
    else:
        # Rolling approach
        df = binance_trades.copy()
        df['rolling_mean'] = df['qty'].rolling(rolling_window).mean()
        df['rolling_std'] = df['qty'].rolling(rolling_window).std()
        df.dropna(subset=['rolling_mean', 'rolling_std'], inplace=True)

        threshold = df['rolling_mean'] + rolling_multiplier * df['rolling_std']
        sig = df['qty'] > threshold
        sig_trades = df[sig]
        logging.info(f"Identified {len(sig_trades)} significant trades via rolling threshold approach.")
        return sig_trades


def align_trades_to_lob(
    sig_trades: pd.DataFrame,
    lob_data: pd.DataFrame,
    time_offset_ms: float = 100
) -> pd.DataFrame:
    """
    Aligns significant trades to LOB snapshots.
    
    Parameters:
    -----------
    sig_trades : pd.DataFrame
        DataFrame with significant trades
    lob_data : pd.DataFrame
        DataFrame with LOB data
    time_offset_ms : float
        Time offset to apply when aligning trades to LOB snapshots (in milliseconds)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with significant trades aligned to LOB snapshots
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{len(sig_trades)}_{len(lob_data)}_{time_offset_ms}"
    cache_file = f"{cache_dir}/aligned_trades_{hash(cache_key)}.pkl"

    if os.path.exists(cache_file):
        logging.info(f"Loading aligned trades from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    if sig_trades.empty or lob_data.empty:
        return pd.DataFrame()

    sig_trades_sorted = sig_trades.sort_index()
    shifted_times = sig_trades_sorted.index + pd.Timedelta(milliseconds=time_offset_ms)

    used_idx = set()
    aligned_indices = []
    aligned_timestamps = []
    valid_mask = []

    for t in shifted_times:
        idx = lob_data.index.searchsorted(t, side='left')
        while idx < len(lob_data) and idx in used_idx:
            idx += 1

        if idx < len(lob_data):
            aligned_indices.append(idx)
            aligned_timestamps.append(lob_data.index[idx])
            used_idx.add(idx)
            valid_mask.append(True)
        else:
            aligned_indices.append(None)
            aligned_timestamps.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    sig_trades_aligned = sig_trades_sorted[valid_mask].copy()
    sig_trades_aligned['aligned_index'] = np.array(aligned_indices)[valid_mask]
    sig_trades_aligned['aligned_timestamp'] = np.array(aligned_timestamps)[valid_mask]

    logging.info(f"Aligned {len(sig_trades_aligned)} trades to LOB snapshots")
    sig_trades_aligned.to_pickle(cache_file)
    logging.info(f"Cached aligned trades to: {cache_file}")

    return sig_trades_aligned