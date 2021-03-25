import math
import more_itertools
import numpy as np
import pandas as pd
import sys

from gourdian.gpd import gpd
from gourdian.lib import errors as lib_errors
from gourdian.utils import pdutils
from gourdian.utils import strutils


_UNSET = object()

COERCE = lib_errors.COERCE


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

  def clone(self, target=_UNSET, head_bombs=_UNSET):
    return SequenceFilter(
      target=pdutils.coalesce(target, self.target, unset=_UNSET),
      head_bombs=pdutils.coalesce(head_bombs, self.head_bombs, unset=_UNSET),
    )

  @property
  def target(self):
    return self._target

  @property
  def head_bombs(self):
    return self._head_bombs


class QueryAnd:
  def __init__(self, sequence_filters):
    self._sequence_filters = tuple(sequence_filters)

  def __repr__(self):
    filts = {f.target.name: f.head_bombs for f in self.sequence_filters}
    return '<%s %r>' % (self.__class__.__name__, filts)

  @property
  def sequence_filters(self):
    return self._sequence_filters

  def match_sequence_filters(self, layout):
    for filt in self.sequence_filters:
      try:
        target = filt.target.bind_layout(layout=layout)
        yield filt.clone(target=target)
      except lib_errors.NoLabelColumnsError:
        pass

  def match_ok_chunk_mask(self, layout):
    ok_chunk_mask = np.ones(shape=layout.sequence.shape, dtype=bool)
    assert ok_chunk_mask.shape == layout.sequence.shape
    for filt in self.match_sequence_filters(layout=layout):
      label_column = filt.target.label_column
      column_index = label_column.column_index
      seq = label_column.sequence
      # Disable everything in ok_chunk_mask that does not match filt.
      filt_bad_mask = np.ones(shape=seq.shape, dtype=bool)
      for head, bomb in filt.head_bombs:
        head_index = seq.left_index_of(value=head, errors=COERCE)
        bomb_index = seq.right_index_of(value=bomb, errors=COERCE)
        filt_bad_mask[slice(head_index, bomb_index)] = False
      filt_bad_selector = ([slice(None)] * column_index) + [filt_bad_mask]
      ok_chunk_mask[tuple(filt_bad_selector)] = False
    return ok_chunk_mask


class Query:
  def __init__(self, subqueries):
    self._subqueries = subqueries

  def __repr__(self):
    return '<%s subqueries=%r>' % (self.__class__.__name__, self.subqueries)

  def describe(self, f=sys.stdout):
    subquery_parts = []
    for subquery in self.subqueries:
      subquery_parts.append('- OR')
      for filt in subquery.sequence_filters:
        subquery_parts.append('  + %s' % (filt.target.name,))
        subquery_parts.extend('    - [%r ... %r] (len=%r)' % (h, b, b-h)
                              for h, b in filt.head_bombs)
    subquery_str = '\n'.join(p for p in subquery_parts if p is not None)
    parts = [
      '# Query (%d subqueries)' % (len(self.subqueries),),
      subquery_str,
    ]
    ret = '\n'.join(p for p in parts if p is not None)
    if not f:
      return ret
    f.write(ret)
    f.write('\n')
    f.flush()

  @property
  def subqueries(self):
    return self._subqueries

  def match_ok_chunk_mask(self, layout):
    ok_chunk_mask = np.zeros(shape=layout.sequence.shape, dtype=bool)
    for subquery in self.subqueries:
      subquery_ok_chunk_mask = subquery.match_ok_chunk_mask(layout=layout)
      ok_chunk_mask = np.logical_or(ok_chunk_mask, subquery_ok_chunk_mask)
    return ok_chunk_mask

  def layout_match(self, layout):
    ok_chunk_mask = self.match_ok_chunk_mask(layout=layout)
    return LayoutMatch(query=self, layout=layout, ok_chunk_mask=ok_chunk_mask)

  def all_layout_matches(self, layouts, fetch_all_stats=True):
    for layout in layouts:
      layout_match = self.layout_match(layout=layout)
      layout_match.fetch_all_stats()
      yield layout_match

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
    layout_matches = tuple(self.all_layout_matches(layouts=layouts, fetch_all_stats=False))
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
  def __init__(self, query, layout, ok_chunk_mask):
    self._query = query
    self._layout = layout
    self._ok_chunk_mask = ok_chunk_mask
    # Placeholders.
    self._matched_chunk_mask = None
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
      'rows=%s' % (strutils.format_number(num_rows, units=strutils.NUMBER_UNITS),),
      'csv_bytes=%s' % (strutils.format_number(csv_bytes, units=strutils.SI_UNITS),),
      'gz_bytes=%s' % (strutils.format_number(gz_bytes, units=strutils.SI_UNITS),),
    ]
    return '<%s %r %s>' % (self.__class__.__name__, str(self.layout),
                           ' '.join(x for x in parts if x))

  def describe(self, f=sys.stdout, max_chunks=10):
    # Stats.
    num_rows_str = strutils.format_number(self.matched_num_rows())
    num_chunks_str = strutils.format_number(self.matched_num_chunks())
    csv_bytes_str = strutils.format_number(self.matched_csv_bytes(), units=strutils.BYTE_UNITS)
    gz_bytes_str = strutils.format_number(self.matched_gz_bytes(), units=strutils.BYTE_UNITS)
    # Chunks.
    matched_chunks = self.matched_chunks()
    head_max_chunks = math.ceil(max_chunks / 2)
    tail_max_chunks = math.floor(max_chunks / 2)
    tail_max_chunks = min(tail_max_chunks, max(0, len(matched_chunks) - head_max_chunks))
    chunks_parts = [
      '\n'.join(['- %s (%d rows)' % (x.filename, x.num_rows())
                 for x in matched_chunks[:head_max_chunks]]),
      '...' if len(matched_chunks) > max_chunks else None,
      '\n'.join(['- %s (%d rows)' % (x.filename, x.num_rows())
                 for x in matched_chunks[-tail_max_chunks:]])
    ]
    chunks_parts = [p for p in chunks_parts if p]
    chunks_str = '\n'.join(chunks_parts)
    # Layout steps.
    label_column_parts = ['- %s: %s' % (x.name, x.labeler()) for x in self.layout.label_columns]
    # Assemble.
    parts = [
      '# LayoutMatch',
      'Endpointer: `%s`' % (self.layout.endpointer,),
      '',
      '## Stats',
      'Matched: %s rows (across %s chunks)' % (num_rows_str, num_chunks_str),
      'Filesize: %s (%s uncompressed)' % (gz_bytes_str, csv_bytes_str),
      '',
      '## Matched Label Columns (%d)' % (len(self.layout.label_columns),),
      '\n'.join(label_column_parts) if label_column_parts else None,
      '',
      '## Chunks (%d)' % (self.matched_num_chunks(),),
      chunks_str or None,
    ]
    ret = '\n'.join(p for p in parts if p is not None)
    if not f:
      return ret
    f.write(ret)
    f.write('\n')
    f.flush()

  @property
  def client(self):
    return self.layout.client

  @property
  def query(self):
    return self._query

  @property
  def layout(self):
    return self._layout

  @property
  def ok_chunk_mask(self):
    return self._ok_chunk_mask

  @property
  def matched_chunk_mask(self):
    if self._matched_chunk_mask is None:
      # Filter out hit chunks that contain 0 rows.
      array_num_rows = self.layout.array(name='num_rows').values()
      self._matched_chunk_mask = np.logical_and(self.ok_chunk_mask, array_num_rows)
    return self._matched_chunk_mask

  def matched_num_chunks(self):
    if self._matched_num_chunks is None:
      self._matched_num_chunks = self.matched_chunk_mask.sum()
    return self._matched_num_chunks

  def matched_num_rows(self):
    if self._matched_num_rows is None:
      array_num_rows = self.layout.array(name='num_rows').values()
      self._matched_num_rows = array_num_rows[self.matched_chunk_mask].sum()
    return self._matched_num_rows

  def matched_csv_bytes(self):
    if self._matched_csv_bytes is None:
      array_csv_bytes = self.layout.array(name='csv_bytes').values()
      self._matched_csv_bytes = array_csv_bytes[self.matched_chunk_mask].sum()
    return self._matched_csv_bytes

  def matched_gz_bytes(self):
    if self._matched_gz_bytes is None:
      array_gz_bytes = self.layout.array(name='gz_bytes').values()
      self._matched_gz_bytes = array_gz_bytes[self.matched_chunk_mask].sum()
    return self._matched_gz_bytes

  def fetch_all_stats(self):
    return {
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
    bucket_indices = np.argwhere(self.matched_chunk_mask)
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
