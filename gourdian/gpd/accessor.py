import pandas as pd

from gourdian.gpd import gpd
from gourdian.lib import lib


@pd.api.extensions.register_dataframe_accessor('gourdian')
class GourdianAccessor:
  def __init__(self, df):
    self._df = df

  ##################################################################################################
  # GTYPING
  def gtypes(self, how, munge_ok=True):
    return gpd.df_to_gtypes(df=self._df, hows=how, munge_ok=munge_ok)

  ##################################################################################################
  # LAYOUT LABELING
  def buckets(self, how, endpointer=None, munge_ok=True):
    """Coax df to the proper gtypes, then bucket and return"""
    gtypes_df = gpd.df_to_gtypes(df=self._df, hows=how, endpointer=endpointer, munge_ok=False)
    return gpd.gtypes_to_buckets(gtypes_df=gtypes_df, hows=how, endpointer=endpointer,
                                 munge_ok=munge_ok)

  def labels(self, how, endpointer=None, munge_ok=True):
    """Coax df to the proper gtypes, then bucket, then label and return"""
    gtypes_df = gpd.df_to_gtypes(df=self._df, hows=how, endpointer=endpointer, munge_ok=False)
    buckets_df = gpd.gtypes_to_buckets(gtypes_df=gtypes_df, hows=how, endpointer=endpointer,
                                       munge_ok=False)
    return gpd.buckets_to_labels(buckets_df=buckets_df, hows=how, endpointer=endpointer,
                                 munge_ok=munge_ok)

  def filenames(self, how, endpointer=None, gz_ok=True):
    """Coax df to the proper gtypes, then bucket, then label, then filename and return"""
    return gpd.df_to_filenames(df=self._df, hows=how, endpointer=endpointer, gz_ok=gz_ok)

  ##################################################################################################
  # LAYOUT JOINING
  def query(self, how, endpointer=None):
    """Returns Query approximately representing the rows in this dataframe."""
    return gpd.df_to_query(df=self._df, hows=how, endpointer=endpointer)

  def match(self, how, endpointer):
    """Returns a LayoutMatch object representing the best match between df and endpointer."""
    query = self.query(how=how, endpointer=endpointer)
    layouts = gpd._layouts(endpointer=endpointer)
    return query.match_layouts(layouts=layouts)
