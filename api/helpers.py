import gzip
from io import StringIO

import pandas as pd


def decompress_dataframe(compressed_data: bytes) -> pd.DataFrame:
    decompressed_data = gzip.decompress(compressed_data)
    df = pd.read_csv(StringIO(decompressed_data.decode("utf-8")))
    return df
