import itertools
import more_itertools
import numbers
import numpy as np
import pandas as pd
import warnings

from gourdian import api
from gourdian.lib import gtypes
from gourdian.lib import errors as lib_errors
from gourdian.lib import lib
from gourdian.lib import queries
from gourdian.utils import pdutils


_UNSET = object()

_REQUIRE_GTYPE = ('gtype',)
_REQUIRE_LABELER = ('labeler',)


def parse_target(obj):
  if isinstance(obj, str):
    if '.' in obj:
      # Column names may not include '.', so this must be a gtype.
      return GTypeTarget(gtype=gtypes.gtype(qualname=obj))
    else:
      # Column name without '.' must refer to a LabelColumn on some layout.
      return LabelColumnNameTarget(name=obj)
  elif gtypes.is_gtype(obj):
    # GType subclass is a valid target.
    return GTypeTarget(gtype=obj)
  elif isinstance(obj, gtypes.GTypeLabeler):
    # GTypeLabeler is a valid target.
    return GTypeLabelerTarget(labeler=obj)
  elif isinstance(obj, lib.LabelColumn):
    # LabelColumn instance is a valid target type.
    return LabelColumnTarget(label_column=obj)
  elif isinstance(obj, HowTarget):
    # HowTarget instance is already a valid target type!
    return obj
  else:
    raise TypeError('unknown target type: %s %r' % (type(obj), obj,))


def parse_hows(hows, require=(), bind_layout=None, errors=lib_errors.RAISE):
  # TODO(rob): Add support for binding to tables, datasets, and users?
  def parse_how(how, via_column=None):
    if how is None:
      raise TypeError('how is None')
    if isinstance(how, (tuple, list)):
      # Every element [how, how, ...] must/will belong to the same provided via_column.
      return [more_itertools.one(parse_how(how=h, via_column=via_column)) for h in how]
    if isinstance(how, How):
      if via_column is None:
        # Assert that this how already belongs to *some* via_column, since one wasn't given.
        if how.via_column is None:
          raise ValueError('how.via_column is empty: %r' % (how,))
      elif (how.via_column is not None) and (how.via_column != via_column):
        # Assert that this how already belongs to provided via_column.
        raise ValueError('how.via_column does not match %s: %r' % (via_column, how.via_column))
      elif how.via_column is None:
        return [how.clone(via_column=via_column)]
      return [how]
    if via_column:
      # This how is a target that must be converted to a how that belongs to via_column.
      return [How(target=parse_target(how), via_column=via_column)]
    raise TypeError('how has bare target; try how={col_name: target}: %r' % (how,))

  lib_errors.validate_errors(errors=errors, drop_ok=True, raise_ok=True)
  # 1. Provided hows may be specified as {via_column: target/How}, [How, ...], or How.
  if isinstance(hows, dict):
    # dict: must be a mapping of {via_column: how}, which is converted to [How, How, ...] here.
    ret = []
    for via_column, target in hows.items():
      ret.extend(parse_how(how=target, via_column=via_column))
  elif isinstance(hows, (tuple, list)):
    # tuple: must contain How instances which already belong to some via_column.
    for how in hows:
      if not isinstance(how, How):
        raise TypeError('tuple how has bare target; try how={col_name: target}: %r' % (how,))
      if how.via_column is None:
        raise ValueError('how.via_column is empty: %r' % (how,))
    ret = hows
  else:
    # obj: how must be coerceable into How instance that already belongs to some via_column.
    ret = parse_how(how=hows)
  # 2. If provided, bind all how clauses to a given layout.
  if bind_layout:
    ret = [h.bind_layout(layout=bind_layout, errors_missing=errors) for h in ret]
    ret = tuple(x for x in ret if x is not None)
  # 3. Assert that all required attributes are present on all how clauses.
  if require:
    ret = list(ret)
    for i, how in enumerate(ret):
      for require_attr in require:
        if getattr(how.target, require_attr, None) is None:
          if errors == lib_errors.RAISE:
            raise TypeError('cannot extract %s: %r' % (require_attr, how.target))
          if errors == lib_errors.DROP:
            ret[i] = None
            break
    ret = tuple(x for x in ret if x is not None)
  return ret


def fit_snug_hows(hows, layouts, errors_missing=lib_errors.RAISE):
  """Returns Hows bound as tightly as possible to a set of layouts."""
  def how_snug_labeler(how, layouts):
    if how.target.labeler is not None:
      # How.target has a labeler; may be LabelColumnTarget or GTypeLabelerTarget.
      gtype = how.target.gtype
      snug_step = how.target.labeler.step
      snug_head = how.target.labeler.head
    elif layouts:
      # Extract the minimum value of step for this how across all layouts.
      layout_gtypes = set()
      layout_steps = set()
      layout_heads = set()
      for layout in layouts:
        # Bind how to layout if possible; ignore this how if it can't be bound to layout (DROP).
        # This will legitimately happen with the 2 layouts: ('lat_lng', 'year_month').
        bound_how = how.bind_layout(layout=layout, errors_missing=lib_errors.DROP)
        if bound_how is not None:
          layout_gtypes.add(bound_how.target.gtype)
          layout_steps.add(bound_how.target.labeler.step)
          layout_heads.add(bound_how.target.labeler.head)
      if len(layout_gtypes) > 1:
        # Since a GTypeLabeler can label exactly one gtype, crash if we found multiple.
        raise lib_errors.MultipleLabelColumnsError('found %d gtype matches in layouts: %r' %
                                                   (len(layout_gtypes), how))
      if len(layout_gtypes) == 0:
        # A spurious how clause (no matches with any layouts) also cannot create a labeler.
        raise lib_errors.NoLabelColumnsError('found 0 gtype matches in layouts: %r' % (how,))
      # We were provided with at least one layout; use min(step/head) for their labeler.
      gtype = more_itertools.one(layout_gtypes)
      snug_step = min(layout_steps)
      snug_head = min(layout_heads)
    else:
      # With no provided layouts, how.target *must* have a gtype or we can't create a labeler.
      gtype = how.target.gtype
      snug_step = gtype.SANE_STEP
      snug_head = pdutils.coalesce(gtype.HEAD, 0)
    return gtype.labeler(step=snug_step, head=snug_head)

  def fit_how(how, layouts, errors_missing):
    if how.target.labeler is not None:
      # How is already bound to a labeler (GTypeLabelerTarget or LabelColumnTarget); done.
      return how
    # else: How is bound to a gtype/label_column_name; return GTypeLabeler() with the most snug fit.
    try:
      labeler = how_snug_labeler(how=how, layouts=layouts)
      return how.clone(target=labeler)
    except lib_errors.NoLabelColumnsError:
      if errors_missing == lib_errors.DROP:
        return None
      raise

  lib_errors.validate_errors(errors=errors_missing, drop_ok=True, raise_ok=True)
  hows = parse_hows(hows=hows)
  fit_hows = [fit_how(how=h, layouts=layouts, errors_missing=errors_missing) for h in hows]
  return tuple(h for h in fit_hows if h is not None)


def parse_bounds(obj):
  def make_bounds_bumper(lower_bump, upper_bump):
    def to_bounds(values):
      lower_bounds = values + lower_bump
      upper_bounds = values + upper_bump
      return pd.concat((lower_bounds, upper_bounds), axis=1)
    return to_bounds

  if hasattr(obj, '__call__'):
    # A fn is understood to produce a 2-column DataFrame of lower and upper bounds.
    return obj
  if isinstance(obj, (tuple, list)):
    # A tuple is understood to produce (val+bounds[0], val+bounds[1]).
    if len(obj) != 2:
      raise ValueError('bounds iterable must be length 2: %r' % (obj,))
    return make_bounds_bumper(lower_bump=obj[0], upper_bump=obj[1])
  if isinstance(obj, numbers.Number):
    # A single numeric type is the same as giving the tuple (-bounds, +bounds).
    return make_bounds_bumper(lower_bump=-obj, upper_bump=obj)
  raise TypeError('unknown bounds type: %s %r' % (type(obj), obj))


class How:
  """How specifies a mapping between a df column (via_column) and some target (gtype, label col).

  Optional coax_kwargs allow df[via_column] to be transformed during the df->gtype phase; supported
  kwargs are gtype-dependent and are provided to the gtype's coax method as gtype.coax(**kwargs).
  """

  @classmethod
  def from_js(cls, how_js):
    target_js = how_js['target']
    return cls(
      name=how_js['name'],
      target=GTypeTarget(
        gtype=gtypes.gtype(
          qualname=target_js['gtype']['qualname'],
          create_kwargs=target_js['gtype']['create_kwargs'],
        ),
      ),
      via_column=how_js['via_column'],
      coax_kwargs=how_js['coax_kwargs'],
    )

  def to_js(self):
    return {
      'name': self.name,
      'target': {
        'gtype': self.target.gtype.to_js(),
      },
      'via_column': self.via_column,
      'coax_kwargs': self.coax_kwargs,
    }

  def __init__(self, target, via_column=None, name=None, bounds=None, **coax_kwargs):
    self._target = parse_target(obj=target)
    self._via_column = via_column
    self._name = name
    self._bounds = bounds if bounds is None else parse_bounds(obj=bounds)
    self._coax_kwargs = coax_kwargs

  def clone(self, target=_UNSET, via_column=_UNSET, name=_UNSET, bounds=_UNSET, **coax_kwargs):
    kwargs = self.coax_kwargs
    kwargs.update(coax_kwargs)
    return How(
      target=pdutils.coalesce(target, self.target, unset=_UNSET),
      via_column=pdutils.coalesce(via_column, self.via_column, unset=_UNSET),
      name=pdutils.coalesce(name, self._name, unset=_UNSET),
      bounds=pdutils.coalesce(bounds, self.bounds, unset=_UNSET),
      **kwargs,
    )

  def __repr__(self):
    return '<%s {%r: %r}>' % (self.__class__.__name__, self.via_column, self.target)

  def __str__(self):
    return self.name

  def bind_layout(self, layout, errors_missing=lib_errors.RAISE):
    """Returns a clone of this How with its target bound to a specific layout.

    Since the target of a How instance point to a string label colum name, a concrete LabelColumn
    instance, a generic GType class, or a GTypeLabeler instance, some work may be necessary to map
    self.target to a specific layout:
      1. If self.target is GTypeTarget:
        - target.gtype matches one column: new LabelColumnTarget(layout.label_column(gtype))
        - target.gtype matches multiple columns: raise MultipleLabelColumnsError (always)
        - target.gtype matches no columns: raise NoLabelColumnsError (RAISE) or None (DROP)
      2. If self.target is GTypeLabelerTarget:
        - target.labeler matches one column: new LabelColumnTarget(layout.label_column(gtype, label_kwargs))
        - target.labeler matches multiple columns: raise MultipleLabelColumnsError (always)
        - target.labeler matches no columns: raise NoLabelColumnsError (RAISE) or None (DROP)
      3. If self.target is LabelColumnNameTarget:
        - target.name in layout: new LabelColumnTarget(layout.label_column(name))
        - target.name not in layout: raise NoLabelColumnsError (RAISE) or None (DROP)
      4. If self.target is LabelColumnTarget:
        - target.label_column in layout: target unchanged
        - target.label_column not in layout: raise NoLabelColumnsError (RAISE) or None (DROP)
    """
    lib_errors.validate_errors(errors=errors_missing, drop_ok=True, raise_ok=True)
    try:
      return self.clone(target=self.target.bind_layout(layout=layout))
    except lib_errors.NoLabelColumnsError:
      if errors_missing == lib_errors.DROP:
        return None
      raise

  @property
  def name(self):
    if self._name is None:
      return self.target.how_name(how=self)
    return self._name

  @property
  def target(self):
    return self._target

  @property
  def bounds(self):
    return self._bounds

  @property
  def coax_kwargs(self):
    return dict(self._coax_kwargs) if self._coax_kwargs else {}

  @property
  def via_column(self):
    return self._via_column

  def coax(self, obj):
    return self.target.gtype.coax(obj=obj, **self.coax_kwargs)



class HowTarget:
  def __init__(self, name):
    self.name = name

  def bind_layout(self, layout, name=None, gtype=None, label_kwargs=None):
    label_column = layout.label_column(name=name, gtype=gtype, label_kwargs=label_kwargs)
    return LabelColumnTarget(label_column=label_column)

  def how_name(self, how):
    """Returns the value used by How.name for any How instance that points to this target."""
    kw = ((k, v) for k, v in how.coax_kwargs.items() if not pdutils.is_empty(v))
    kw_str = ','.join('%s=%r' % (k, v) for k, v in kw)
    kw_str = '(%s)' % (kw_str,) if kw_str else ''
    return '%s%s|%s' % (how.via_column, kw_str, self.name)


class GTypeTarget(HowTarget):
  def __init__(self, gtype):
    self._gtype = gtype

  def __repr__(self):
    return '<%s gtype=%s>' % (self.__class__.__name__, self.gtype,)

  def __str__(self):
    return repr(self.gtype)

  def bind_layout(self, layout):
    return super().bind_layout(layout=layout, gtype=self.gtype)

  @property
  def name(self):
    return str(self._gtype.qualname())

  @property
  def gtype(self):
    return self._gtype

  @property
  def label_column(self):
    return None

  @property
  def labeler(self):
    # There are no default labeler arguments for gtypes.
    return None


class GTypeLabelerTarget(HowTarget):
  def __init__(self, labeler):
    self._labeler = labeler

  def __repr__(self):
    return '<%s labeler=%s>' % (self.__class__.__name__, self.labeler,)

  def __str__(self):
    return repr(self.labeler)

  def bind_layout(self, layout=None):
    labeler = self._labeler
    return super().bind_layout(layout=layout, gtype=labeler.gtype, label_kwargs=labeler.kwargs)

  @property
  def name(self):
    return self._labeler.name

  @property
  def gtype(self):
    return self._labeler.gtype

  @property
  def label_column(self):
    return None

  @property
  def labeler(self):
    return self._labeler


class LabelColumnNameTarget(HowTarget):
  def __init__(self, name):
    self._name = name

  def __repr__(self):
    return '<%s label_column_name=%s>' % (self.__class__.__name__, self.name,)

  def __str__(self):
    return repr(self.name)

  def bind_layout(self, layout):
    return super().bind_layout(layout=layout, name=self.name)

  @property
  def name(self):
    return self._name

  def how_name(self, how):
    # Returns only the column name, ignoring via_column and coax_kwargs from how.
    return self.name

  @property
  def gtype(self):
    return None

  @property
  def label_column(self):
    return None

  @property
  def labeler(self):
    return None


class LabelColumnTarget(HowTarget):
  def __init__(self, label_column):
    self._label_column = label_column

  def __repr__(self):
    return '<%s label_column=%s>' % (self.__class__.__name__, self.label_column,)

  def __str__(self):
    return repr(self.label_column)

  def bind_layout(self, layout):
    if self.label_column.layout != layout:
      raise lib_errors.NoLabelColumnsError('label_column does not belong to %r: %r' %
                                           (str(layout.endpointer), self.label_column))
    return self

  @property
  def name(self):
    return self._label_column.name

  def how_name(self, how):
    # Returns only the column name, ignoring via_column and coax_kwargs from how.
    return self.name

  @property
  def gtype(self):
    return self._label_column.gtype

  @property
  def label_column(self):
    return self._label_column

  @property
  def labeler(self):
    return self._label_column.labeler()


def _client(endpointer, client=None):
  return client if client else (getattr(endpointer, 'client', None) or api.DEFAULT_CLIENT)


def _layout(endpointer, client=None):
  if endpointer:
    client = _client(endpointer=endpointer, client=client)
    return client.layout(endpointer=endpointer)
  return None


def _layouts(endpointer, client=None):
  if endpointer:
    client = _client(endpointer=endpointer, client=client)
    endpoint = client.endpoint(endpointer=endpointer, require_types=(lib.TABLE, lib.LAYOUT))
    return tuple(endpoint.layouts) if endpoint.endpoint_type == lib.TABLE else (endpoint,)
  return ()


####################################################################################################
# LAYOUT LABELING
def _df_to_lower_upper(df, hows):
  def series_to_lower_upper(values, bounds):
    """Applies bounds to values and returns a 2-tuple of pd.Series of (heads, tails)."""
    if bounds is None:
      return (values, values)
    head_tail_df = bounds(values)
    if len(head_tail_df.columns) != 2:
      raise ValueError('bounds() must produce exactly 2 columns: %r' % (head_tail_df.columns,))
    lowers = head_tail_df.iloc[:, 0]
    uppers = head_tail_df.iloc[:, 1]
    if (lowers > uppers).any():
      mask = lowers > uppers
      raise ValueError('bounds() must produce lower <= upper: %s %r -> %r' %
                       (how.name, values[mask].iloc[0], tuple(head_tail_df[mask].iloc[0])))
    return lowers, uppers

  lowers = []
  uppers = []
  for how in hows:
    lower, upper = series_to_lower_upper(values=df[how.name], bounds=how.bounds)
    lowers.append(lower)
    uppers.append(upper)

  lower_uppers = [series_to_lower_upper(values=df[h.name], bounds=h.bounds) for h in hows]
  lower_df, upper_df = [pd.concat(x, axis=1) for x in zip(*lower_uppers)]
  return lower_df, upper_df


def df_to_gtypes(df, hows, endpointer=None, munge_ok=True, warn_ok=True, client=None):
  layout = _layout(endpointer=endpointer, client=client)
  hows = parse_hows(hows=hows, require=_REQUIRE_GTYPE, bind_layout=layout)
  if warn_ok and any(h.bounds is not None for h in hows):
    warnings.warn('configured bounds will be ignored on all columns; try df_to_gtype_bounds()')
  ret = pd.concat(
    objs=[h.coax(df[h.via_column]) for h in hows],
    axis=1,
  )
  if munge_ok:
    ret.columns = [h.name for h in hows]
  return ret


def df_to_gtype_bounds(df, hows, endpointer=None, munge_ok=True, warn_ok=True, client=None):
  layout = _layout(endpointer=endpointer, client=client)
  hows = parse_hows(hows=hows, require=_REQUIRE_GTYPE, bind_layout=layout)
  gtype_df = df_to_gtypes(df=df, hows=hows, endpointer=endpointer, munge_ok=True, warn_ok=False,
                          client=client)
  gtype_lower_df, gtype_upper_df = _df_to_lower_upper(df=gtype_df, hows=hows)
  if not munge_ok:
    # We created gtype_df with munge_ok=True; reverse that if requested.
    gtype_lower_df.columns = [h.via_column for h in hows]
    gtype_upper_df.columns = [h.via_column for h in hows]
  return gtype_lower_df, gtype_upper_df


def gtypes_to_buckets(gtypes_df, hows, endpointer=None, munge_ok=True, client=None):
  layout = _layout(endpointer=endpointer, client=client)
  hows = parse_hows(hows=hows, require=_REQUIRE_LABELER, bind_layout=layout)
  ret = pd.concat(
    objs=[h.target.labeler.bucket(gtypes_df[h.via_column]) for h in hows],
    axis=1,
  )
  if munge_ok:
    ret.columns = [h.name for h in hows]
  return ret


def buckets_to_labels(buckets_df, hows, endpointer=None, munge_ok=True, client=None):
  layout = _layout(endpointer=endpointer, client=client)
  hows = parse_hows(hows=hows, require=_REQUIRE_LABELER, bind_layout=layout)
  ret = pd.concat(
    objs=[h.target.labeler.label(buckets_df[h.via_column]) for h in hows],
    axis=1,
  )
  if munge_ok:
    ret.columns = [h.name for h in hows]
  return ret


def labels_to_filenames(labels_df, gz_ok=True):
  suffix = '.csv.gz' if gz_ok else '.csv'
  ret = 'chunk|' + labels_df.iloc[:, 0].str.cat(labels_df.iloc[:, 1:], sep='|') + suffix
  return ret.rename('filename')


def _overlaps_range(old, new):
  for (old_head, old_bomb), (new_head, new_bomb), in zip(old, new):
    if (new_head > old_bomb) or (new_bomb < old_head):
      # Either new_head or new_bomb is outside the old range.
      return False
  return True


def _collapse_ranges(lower_df, upper_df):
  assert lower_df.shape == upper_df.shape
  old = None
  for (_, *heads), (_, *bombs) in zip(lower_df.itertuples(), upper_df.itertuples()):
    new = list(list(x) for x in zip(heads, bombs))
    if old is None:
      old = new
    elif _overlaps_range(old=old, new=new):
      # The new range overlaps the old range, but the old range may need to be enlarged.
      for old_hb, new_hb in zip(old, new):
        old_head, old_bomb = old_hb
        new_head, new_bomb = new_hb
        if (new_head < old_head):
          # The new head is smaller than old head, but is within min_step of it; enlarge old.
          old_hb[0] = new_head
        if (new_bomb > old_bomb):
          # The new bomb is larger than old bomb, but is within min_step of it; enlarge old.
          old_hb[1] = new_bomb
    else:
      # The new range does not overlap the old range; emit the old range and keep new as old.
      yield old
      old = new
  # Emit the final range.
  yield old


def _collapse_dfs(lower_df, upper_df):
  # Two ranges can be merged if their values overlap for every column.
  collapsed_lower_df = lower_df
  collapsed_upper_df = upper_df
  # Permute all orderings of columns to ensure we get all possible collapses.
  # TODO(rob): Is this required? How to improve? This is so lazy.
  for lower_cols, upper_cols in zip(itertools.permutations(lower_df.columns),
                                    itertools.permutations(upper_df.columns)):
    collapsed_lower_df = collapsed_lower_df.reindex(columns=lower_cols)
    collapsed_upper_df = collapsed_upper_df.reindex(columns=upper_cols)
    ranges = tuple(_collapse_ranges(lower_df=collapsed_lower_df, upper_df=collapsed_upper_df))
    collapsed_lower_df = pd.DataFrame([[c[0] for c in row] for row in ranges], columns=lower_cols)
    collapsed_upper_df = pd.DataFrame([[c[1] for c in row] for row in ranges], columns=upper_cols)
  # Return dfs back to original column order.
  collapsed_lower_df = collapsed_lower_df.reindex(columns=lower_df.columns)
  collapsed_upper_df = collapsed_upper_df.reindex(columns=upper_df.columns)
  # Recurse if the number of rows was reduced; more reductions may be possible.
  if len(lower_df) > len(collapsed_lower_df):
    # TODO(rob): Why is this recursion step required? Probably a bug...
    return _collapse_dfs(lower_df=collapsed_lower_df, upper_df=collapsed_upper_df)
  return collapsed_lower_df, collapsed_upper_df


def df_to_query(df, hows, endpointer=None, client=None, errors_missing=lib_errors.DROP):
  layouts = _layouts(endpointer=endpointer, client=client)
  # 1. Fit hows as snugly as possible to target layouts.
  hows = parse_hows(hows=hows)
  fit_hows = fit_snug_hows(hows=hows, layouts=layouts, errors_missing=errors_missing)
  # 2. Extract lower and upper gtypes_df, accounting for any bounds present in hows.
  gtypes_lower_df, gtypes_upper_df = df_to_gtype_bounds(df=df, hows=fit_hows)
  # 3. Extract buckets from df using fit_hows, accounting for munged column names in gtypes_df.
  munged_hows = [h.clone(via_column=h.name, name=h.name) for h in fit_hows]
  buckets_lower_df = gtypes_to_buckets(gtypes_df=gtypes_lower_df, hows=munged_hows).add_prefix('<')
  buckets_upper_df = gtypes_to_buckets(gtypes_df=gtypes_upper_df, hows=munged_hows).add_suffix('>')
  assert buckets_lower_df.shape == buckets_upper_df.shape
  # 4. Bump the upper bounds out by step to ensure we don't get any 0-width query conditions.
  buckets_upper_df = buckets_upper_df + [h.target.labeler.step for h in fit_hows]
  # 5. De-dupe bucket ranges by concatenating lower+upper and dropping dupes, and sort by heads.
  unique_range_df = pd.concat((buckets_lower_df, buckets_upper_df), axis=1).drop_duplicates()
  unique_range_df = unique_range_df.sort_values(by=list(buckets_lower_df.columns))
  unique_lower_df = unique_range_df.iloc[:, :len(buckets_lower_df.columns)]
  unique_upper_df = unique_range_df.iloc[:, len(buckets_lower_df.columns):]
  # 6. Merge sorted ranges into contiguous ranges spaced at least min_step apart.
  lower_df, upper_df = _collapse_dfs(lower_df=unique_lower_df, upper_df=unique_upper_df)
  # 7. Convert ranges to QueryAnd subqueries.
  subqueries = []
  for (_, *heads), (_, *bombs) in zip(lower_df.itertuples(), upper_df.itertuples()):
    sequence_filters = []
    for how, head, bomb in zip(hows, heads, bombs):
      filt = queries.SequenceFilter(
        target=how.target,
        head_bombs=[(head, bomb)],
      )
      sequence_filters.append(filt)
    subqueries.append(queries.QueryAnd(sequence_filters=sequence_filters))
  # 8. Return a Query made up of all subqueries.
  return queries.Query(subqueries=subqueries)
