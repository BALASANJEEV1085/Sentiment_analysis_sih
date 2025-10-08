import pandas as pd
from io import BytesIO

def load_uploaded_file(uploaded_file):
    """
    load CSV / Excel / JSON into a pandas DataFrame
    uploaded_file is a Streamlit uploaded file object
    """
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    bytes_data = uploaded_file.read()
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(BytesIO(bytes_data))
        elif name.endswith('.xlsx') or name.endswith('.xls'):
            df = pd.read_excel(BytesIO(bytes_data))
        elif name.endswith('.json'):
            # try orient='records' fallback
            try:
                df = pd.read_json(BytesIO(bytes_data), orient='records', lines=False)
            except ValueError:
                df = pd.read_json(BytesIO(bytes_data), orient='records', lines=True)
        else:
            raise ValueError("Unsupported file type. Supported: CSV, Excel, JSON.")
    except Exception as e:
        raise e
    return df
