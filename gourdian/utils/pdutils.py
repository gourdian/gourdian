import datetime
import json
import numpy as np
import pandas as pd


class JSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.int64):
      return int(obj)
    if isinstance(obj, datetime.datetime):
      return obj.isoformat()
    return json.JSONEncoder.default(self, obj)


def json_dumps(obj):
  return json.dumps(obj, cls=JSONEncoder)


def is_empty(val, unset=None):
  return pd.isnull(val) or ((not val) and val != 0) or (val is unset)


def coalesce(*vals, unset=None):
  for val in vals:
    if not is_empty(val, unset=unset):
      return val
  # Return last item if everything is empty, to let caller set a default; e.g. coalesce(None, {}).
  return val


def concat_reindex_old(df, index_df):
  # Glue index_df onto the front of df, joining by index.
  merged_df = pd.concat([index_df, df], axis=1)
  # Re-index merged_df using the index_df columns glued on to the front.
  index_cols = [merged_df.iloc[:, i] for i in range(len(index_df.columns))]
  merged_df.set_index(index_cols, inplace=True)
  # Drop the index_df columns that have been promoted to the index of merged_df.
  return merged_df.iloc[:, len(index_cols):]


def concat_reindex(df, index_df):
  # Glue index_df onto the front of df, joining by index (allowing for sorting differences).
  merged_df = pd.concat([index_df, df], axis=1)
  # Create a MultiIndex explicitly, because we always want a multiindex (even a 1-col multiindex).
  index_cols = [merged_df.columns[i] for i in range(len(index_df.columns))]
  new_index = pd.MultiIndex.from_frame(merged_df[index_cols])
  merged_df.index = new_index
  # Drop the index_df columns that have been promoted to the index of merged_df.
  return merged_df.iloc[:, len(index_cols):]


def nullify_na(df, inplace=True):
  """Like df.dropna(), but rows are replaced with all None values instead of being removed.

  >>> df = pd.DataFrame({'age': [1, None, 3], 'price': [None, 20, 30]})
  >>> nullify_na(df=df)
     age  price
  0  NaN    NaN
  1  NaN    NaN
  2  3.0   30.0
  """
  if not inplace:
    df = df.copy()
  df.loc[df.isnull().any(axis=1), :] = None
  return df
