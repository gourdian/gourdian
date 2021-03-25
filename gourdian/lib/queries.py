import more_itertools
import numpy as np
import pandas as pd
import sys

from gourdian.gpd import gpd
from gourdian.lib import errors as lib_errors
from gourdian.utils import pdutils


def parse_sequence_filters():
  pass


class SequenceFilter:
  def __init__(self, target, head_bombs=()):
    self._target = gpd.parse_target(obj=target)
    self._head_bombs = tuple(tuple(x) for x in head_bombs)

  def __repr__(self):
    return '<%s %r in %r>' % (self.__class__.__name__, self.target, self.head_bombs)

  def __len__(self):
    return len(self._head_bombs)

  def __iter__(self):
    return iter(self._head_bombs)

  @property
  def target(self):
    return self._target

  @property
  def head_bombs(self):
    return self._head_bombs


class Query:
  def __init__(self, sequence_filters):
    self._sequence_filters = tuple(sequence_filters)

  def __repr__(self):
    hb_str = ' '.join('%s=%r' % (filt.target.name, filt._head_bombs)
                      for filt in self._sequence_filters)
    return '<%s %s>' % (self.__class__.__name__, hb_str)

  def describe(self, f=sys.stdout):
    filt_parts = []
    for filt in self.sequence_filters:
      filt_parts.append('- %s (%d ranges)' % (filt.target.name, len(filt.head_bombs)))
      filt_parts.extend(['  - (%r ... %r)' % (h, b) for h, b in filt.head_bombs])
    parts = [
      '# Query Sequence Filters (%d)' % (len(self.sequence_filters),),
      '\n'.join(filt_parts),
    ]
    ret = '\n'.join(p for p in parts if p is not None)
    if not f:
      return f
    f.write(ret)
    f.flush()

  @property
  def sequence_filters(self):
    return self._sequence_filters

  def layout_match(self, layout):
    return LayoutMatch(query=self, layout=layout)

  def layout_matches(self, layouts):
    return tuple(self.layout_match(layout=x) for x in layouts)

  def match_layouts(self, layouts):
    """Returns a LayoutMatch for the most efficient way to satisfy this query against some table.

    This function iteratively discards layouts until only one remains:
    1. If > 1 layouts remain, discard all that require more than the min number of rows
    2. If > 1 layouts remain, discard all that require more than the min number of csv_bytes
    3. If > 1 layouts remain, discard all that require more than the min number of gz_bytes
    4. If > 1 layouts remain, discard all that match fewer than the max number of columns in query
    5. If > 1 layouts remain, discard all but the first one sorted alphabetically by layout name
    """
    # 0. If there is only 1 layout, it is automatically the best.
    layout_matches = self.layout_matches(layouts=layouts)
    if len(layout_matches) < 2:
      return more_itertools.first(layout_matches, default=None)
    # 1. Keep plans that need the fewest rows.
    min_num_rows = min(x.matched_num_rows() for x in layout_matches)
    layout_matches = [x for x in layout_matches if x.matched_num_rows() == min_num_rows]
    if len(layout_matches) == 1:
      return layout_matches[0]
    # 2. Keep plans that need the fewest csv_bytes.
    min_csv_bytes = min(x.matched_csv_bytes() for x in layout_matches)
    layout_matches = [x for x in layout_matches if x.matched_csv_bytes() == min_csv_bytes]
    if len(layout_matches) == 1:
      return layout_matches[0]
    # 3. Keep plans that need the fewest gz_bytes.
    min_gz_bytes = min(x.matched_gz_bytes() for x in layout_matches)
    layout_matches = [x for x in layout_matches if x.matched_gz_bytes() == min_gz_bytes]
    if len(layout_matches) == 1:
      return layout_matches[0]
    # 4. Keep plans that match the greatest number of columns.
    max_cols_matched = max(x.matched_num_columns() for x in layout_matches)
    layout_matches = [x for x in layout_matches if x.matched_num_columns() == max_cols_matched]
    if len(layout_matches) == 1:
      return layout_matches[0]
    # 5. Return the first layout sorted alphabetically.
    return more_itertools.first(sorted(layout_matches, key=lambda m: m.layout.name))

  def match_table(self, table):
    layouts = table.layouts
    return self.match_layouts(layouts=layouts)


class LayoutMatch:
  def __init__(self, query, layout):
    self._query = query
    self._layout = layout
    # Placeholders.
    self._ok_chunk_mask = None
    self._matched_sequence_filters = None
    self._matched_num_columns = None
    self._matched_num_chunks = None
    self._matched_num_rows = None
    self._matched_csv_bytes = None
    self._matched_gz_bytes = None

  def __repr__(self):
    # NOTE: Do not change this to use any values that require additional arrays to be fetched!
    num_chunks = pdutils.coalesce(self._matched_num_chunks, '?')
    num_rows = pdutils.coalesce(self._matched_num_rows, '?')
    csv_bytes = pdutils.coalesce(self._matched_csv_bytes, '?')
    gz_bytes = pdutils.coalesce(self._matched_gz_bytes, '?')
    parts = [
      'chunks=%s' % (num_chunks,),
      'rows=%s' % (num_rows,),
      'csv_bytes=%s' % (csv_bytes,),
      'gz_bytes=%s' % (gz_bytes,),
      'cols=%d' % (self.matched_num_columns(),),
      repr([x.target.name for x in self.matched_sequence_filters()]),
    ]
    return '<%s %r %s>' % (self.__class__.__name__, str(self.layout),
                           ' '.join(x for x in parts if x))

  @property
  def client(self):
    return self.layout.client

  @property
  def query(self):
    return self._query

  @property
  def layout(self):
    return self._layout

  def ok_chunk_mask(self):
    if self._ok_chunk_mask is None:
      layout = self.layout
      # Each sequence_filter mask is combined with AND against an initial "all" mask.
      ok_chunk_mask = np.ones(shape=layout.sequence.shape, dtype=bool)
      for sequence_filter in self.matched_sequence_filters():
        # Dropout parts of ok_chunk_mask that do not overlap with this sequence_filter.
        target = sequence_filter.target.bind_layout(layout=self.layout)
        label_column = target.label_column
        column_index, seq = label_column.column_index, label_column.sequence
        # Each (head, bomb) mask is combined with OR against an initial "none" mask.
        ok_filt_mask = np.zeros(shape=ok_chunk_mask.shape, dtype=bool)
        for head, bomb in sequence_filter.head_bombs:
          head_index = seq.left_index_of(value=head, errors='coerce')
          bomb_index = seq.right_index_of(value=bomb, errors='coerce')
          selector = tuple([slice(None)] * column_index + [slice(head_index, bomb_index)])
          ok_filt_mask[selector] = True
        ok_chunk_mask = np.logical_and(ok_chunk_mask, ok_filt_mask)
      # Filter out hit chunks that contain 0 rows.
      array_num_rows = layout.array(name='num_rows').values()
      self._ok_chunk_mask = np.logical_and(ok_chunk_mask, array_num_rows)
    return self._ok_chunk_mask

  def matched_sequence_filters(self):
    if self._matched_sequence_filters is None:
      ret = []
      for filt in self.query.sequence_filters:
        try:
          _ = filt.target.bind_layout(layout=self.layout)
          ret.append(filt)
        except lib_errors.NoLabelColumnsError:
          pass
      self._matched_sequence_filters = tuple(ret)
    return self._matched_sequence_filters

  def matched_num_columns(self):
    if self._matched_num_columns is None:
      self._matched_num_columns = len(self.matched_sequence_filters())
    return self._matched_num_columns

  def matched_num_chunks(self):
    if self._matched_num_chunks is None:
      ok_mask = self.ok_chunk_mask()
      self._matched_num_chunks = ok_mask.sum()
    return self._matched_num_chunks

  def matched_num_rows(self):
    if self._matched_num_rows is None:
      ok_mask = self.ok_chunk_mask()
      array_num_rows = self.layout.array(name='num_rows').values()
      self._matched_num_rows = array_num_rows[ok_mask].sum()
    return self._matched_num_rows

  def matched_csv_bytes(self):
    if self._matched_csv_bytes is None:
      ok_mask = self.ok_chunk_mask()
      array_csv_bytes = self.layout.array(name='csv_bytes').values()
      self._matched_csv_bytes = array_csv_bytes[ok_mask].sum()
    return self._matched_csv_bytes

  def matched_gz_bytes(self):
    if self._matched_gz_bytes is None:
      ok_mask = self.ok_chunk_mask()
      array_gz_bytes = self.layout.array(name='gz_bytes').values()
      self._matched_gz_bytes = array_gz_bytes[ok_mask].sum()
    return self._matched_gz_bytes

  def fetch_all_stats(self):
    return {
      'columns_matched': [x.target for x in self.matched_sequence_filters()],
      'matched_num_columns': self.matched_num_columns(),
      'matched_num_chunks': self.matched_num_chunks(),
      'matched_num_rows': self.matched_num_rows(),
      'matched_csv_bytes': self.matched_csv_bytes(),
      'matched_gz_bytes': self.matched_gz_bytes(),
    }

  def matched_buckets_df(self):
    """Returns a DataFrame of every unique bucket value required by this layout match.

    If all buckets are required, this method is more performant than iterating
    self.matched_chunks().
    """
    ok_mask = self.ok_chunk_mask()
    bucket_indices = np.argwhere(ok_mask)
    seq_steps, seq_heads = zip(*[(seq.step, seq.head) for seq in self.layout.sequence.sequences])
    buckets = (bucket_indices * seq_steps) + seq_heads
    return pd.DataFrame(buckets, columns=[x.name for x in self.layout.label_columns])

  def matched_labels_df(self):
    """Returns a DataFrame of every unique label value required by this layout match.

    If all labels are required, this method is more performant than iterating self.matched_chunks().
    """
    buckets_df = self.matched_buckets_df()
    return gpd.buckets_to_labels(
      buckets_df=buckets_df,
      hows=[gpd.How(target=c, via_column=c.name) for c in self.layout.label_columns],
    )

  def matched_filenames(self):
    """Returns an iterable of every unique chunk filename required by this layout match.

    If all chunk filenames are required, this method is more performant than iterating
    self.matched_chunks().
    """
    labels_df = self.matched_labels_df()
    return gpd.labels_to_filenames(labels_df=labels_df)

  def matched_chunks(self):
    """Returns a tuple of every LayoutChunk required by this layout match."""
    buckets_df = self.matched_buckets_df()
    layout = self.layout
    ret = []
    for i, *bucket in buckets_df.itertuples():
      ret.append(layout.chunk(bucket=tuple(bucket)))
    return tuple(ret)

  def dfs(self, page_size=None):
    """Returns iterator of DataFrames containing layout rows that satisfies this match.

    May perform an HTTP fetch for chunk csvs that are not cached locally.

    If provided, `page_size` determines the number of rows that are yielded in each DataFrame,
    merging rows from small chunks and splitting rows from large chunks as required.  If unset, one
    DataFrame per chunk is yielded.
    """
    client = self.client
    endpointer = self.layout.endpointer
    if page_size is None:
      # Yields one dataframe per chunk filename.
      for filename in self.matched_filenames():
        yield client.chunk_df(endpointer=endpointer, filename=filename)
    else:
      # Yields dataframes of page_size number of rows.
      leftover_df = None
      for filename in self.matched_filenames():
        first_page_size = 0 if leftover_df is None else (page_size - len(leftover_df))
        page_dfs = client.chunk_dfs(endpointer=endpointer, filename=filename, page_size=page_size,
                                    first_page_size=first_page_size)
        for page_df in page_dfs:
          if leftover_df is not None:
            page_df = pd.concat((leftover_df, page_df))
            leftover_df = None
            assert len(page_df) <= page_size
          if len(page_df) < page_size:
            leftover_df = page_df
          else:
            yield page_df
      if leftover_df is not None:
        yield leftover_df

  def df(self):
    """Returns a single DataFrame containing layout rows that satisfies this match.

    May perform an HTTP fetch for chunk csvs that are not cached locally.
    """
    return pd.concat(self.dfs())
