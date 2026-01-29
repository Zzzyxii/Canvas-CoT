import pickle
import pandas as pd
import os

pkl_path = 'outputs/Gemini-2.5-Pro-Blackboard/T20251227_Gdd03fd1a/Gemini-2.5-Pro-Blackboard_RBench_V_supp.pkl'

if os.path.exists(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
            # Check if it looks like results
            if 'prediction' in data or 'score' in data:
                print("Found prediction/score data.")
            else:
                # Maybe it's a dict of index -> result
                first_key = next(iter(data))
                print(f"First key: {first_key}, Value type: {type(data[first_key])}")
                print(f"First value: {data[first_key]}")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"First item: {data[0]}")
        elif isinstance(data, pd.DataFrame):
            print(data.head())
            print(data.columns)
    except Exception as e:
        print(f"Error loading pickle: {e}")
else:
    print(f"File not found: {pkl_path}")
