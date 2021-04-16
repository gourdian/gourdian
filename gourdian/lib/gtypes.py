import collections.abc
import datetime
import math
import numpy as np
import pandas as pd
import pandas.api.types as pd_types

from gourdian.utils import pdutils


COERCE = 'coerce'


def is_scalar(val):
  if isinstance(val, str):
    return True
  return not isinstance(val, collections.abc.Iterable)


def to_numeric(vals):
  if len(vals) > 0:
    # Work-around for https://github.com/modin-project/modin/issues/2833
    return pd.to_numeric(vals, errors=COERCE)
  return vals


def rename(series, suffix=None, default_series=None):
  name = series.name or getattr(default_series, 'name', None)
  if name and suffix:
    return series.rename('%s:%s' % (name, suffix))
  if name:
    return series.rename(name)
  if suffix:
    return series.rename(suffix)
  return series


def super_gtype(name):
  """Returns the SuperGType class with a given name."""
  return GTypeMeta.super_gtype(name=name)


def gtype(qualname):
  """Returns the GType class with a given qualname; e.g. <SuperGType.name>.<GType.name>."""
  return GTypeMeta.gtype(qualname=qualname)


def all_gtypes():
  """Returns all defined GType classes."""
  return GTypeMeta.all_gtypes()


def is_gtype(value):
  return isinstance(value, type) and issubclass(value, GType)


class GTypeMeta(type):
  """Metaclass for GType/SuperGType that maintains a directory of all gtypes."""
  _SUPER_GTYPES = {}
  _GTYPES = {}

  def __new__(cls, name, bases, classdict):
    klass = type.__new__(cls, name, bases, dict(classdict))
    if name in ('SuperGType', 'GType', 'GTypeNumeric') or name.startswith('_'):
      # Do not register the base types.
      pass
    elif issubclass(klass, SuperGType):
      klass.name = klass.__qualname__.rsplit('.', 1)[-1]
      cls._SUPER_GTYPES[klass.name] = klass
      # Associate all known gtypes that belong to this super_gtype.
      klass.gtypes = cls.gtypes_of(super_gtype=klass)
      for gtype in klass.gtypes:
        gtype.super_gtype = klass
    elif issubclass(klass, GType):
      klass.qualname = '.'.join(klass.__qualname__.rsplit('.', 2)[-2:])
      klass.name = klass.qualname.split('.')[-1]
      cls._GTYPES[klass.qualname] = klass
      # If the super_gtype has already been created, associate it now.
      super_gtype_name = klass.qualname.split('.', 1)[0]
      if super_gtype_name in cls._SUPER_GTYPES:
        super_gtype = cls.super_gtype(name=super_gtype_name)
        klass.super_gtype = super_gtype
        super_gtype.gtypes = cls.gtypes_of(super_gtype=super_gtype)
    else:
      raise TypeError('cannot register type: %r' % (name,))
    return klass

  def __repr__(cls):
    if issubclass(cls, GType):
      return '<gtype %s>' % (cls.qualname,)
    return '<super_gtype %s>' % (cls.name,)

  def __str__(cls):
    if issubclass(cls, GType):
      return cls.qualname
    return cls.name

  @classmethod
  def super_gtype(cls, name):
    return cls._SUPER_GTYPES[name]

  @classmethod
  def gtype(cls, qualname):
    return cls._GTYPES[qualname]

  @classmethod
  def all_gtypes(cls):
    return tuple(cls._GTYPES.values())

  @classmethod
  def gtypes_of(cls, super_gtype):
    super_gtype_name = super_gtype if isinstance(super_gtype, str) else super_gtype.name
    key_prefix = '{}.'.format(super_gtype_name)
    return tuple(sorted((t for k, t in cls._GTYPES.items() if k.startswith(key_prefix)),
                        key=lambda x: x.__name__))


class SuperGType(metaclass=GTypeMeta):
  @classmethod
  def gtype(cls, name):
    qualname = '{}.{}'.format(cls.__name__, name)
    return GTypeMeta.gtype(qualname)

  def __str__(self):
    return self.name


class GType(metaclass=GTypeMeta):
  @classmethod
  def _to_series(cls, val):
    if isinstance(val, pd.Series):
      series = val
    elif isinstance(val, pd.DataFrame):
      raise ValueError("cannot convert DataFrame to Series; try df['col_name']")
    else:
      series = pd.Series(val)
    return series.rename(series.name or cls.qualname)


class GTypeLabeler:
  def __init__(self, gtype, step, head, **extra_kwargs):
    self._gtype = gtype
    self._step = step
    self._head = head
    self._extra_kwargs = extra_kwargs
    self._kwargs = dict(step=step, head=head, **(extra_kwargs or {}))
    # Runtime state.
    self._name = None

  def clone(self, gtype=None, **kwargs):
    new_gtype = self.gtype if gtype is None else gtype
    new_kwargs = self.kwargs
    new_kwargs.update(kwargs)
    return type(self)(gtype=new_gtype, **new_kwargs)

  def __repr__(self):
    kw = ((k, v) for k, v in self._kwargs.items() if not pdutils.is_empty(v))
    kw_str = ', '.join('%s=%r' % (k, v) for k, v in kw)
    return ('<%s %s(%s)>' % (self.__class__.__name__, self._gtype.qualname, kw_str))

  def __str__(self):
    return self.name

  @property
  def name(self):
    if self._name is None:
      kw = ((k, v) for k, v in self._kwargs.items() if not pdutils.is_empty(v))
      kw_str = ', '.join('%s=%r' % (k, v) for k, v in kw)
      self._name = '%s(%s)' % (self._gtype.qualname, kw_str)
    return self._name

  @property
  def gtype(self):
    return self._gtype

  @property
  def head(self):
    return self._head

  @property
  def step(self):
    return self._step

  @property
  def extra_kwargs(self):
    return dict(self._extra_kwargs)

  @property
  def kwargs(self):
    return dict(self._kwargs)

  def bucket(self, val):
    return self._gtype.bucket(val=val, **self._kwargs)

  def label(self, bucket):
    return self._gtype.label(bucket=bucket)


class GTypeNumeric(GType):
  # BOMB/TAIL are mutually exclusive: valid if val < BOMB or val <= TAIL.
  HEAD = None
  BOMB = None
  TAIL = None
  # Format string to use when converting a numeric bucket into a string label.
  LABEL_FMT = '%0.04f'
  # Step size to use when building a Query object with no pre-defined step info.
  SANE_STEP = 1

  @classmethod
  def _is_valid_scalar(cls, val):
    if isinstance(val, (int, float)):
      has_head = cls.HEAD is not None
      has_bomb = cls.BOMB is not None
      has_tail = cls.TAIL is not None
      if has_head and has_bomb:
        return cls.HEAD <= val < cls.BOMB
      if has_head and has_tail:
        return cls.HEAD <= val <= cls.TAIL
      if has_head:
        return cls.HEAD <= val
      if has_bomb:
        return val < cls.BOMB
      if has_tail:
        return val <= cls.TAIL
      return True
    return False

  @classmethod
  def _is_valid_pd(cls, val):
    vals = cls._to_series(val)
    if pd_types.is_numeric_dtype(vals):
      has_head = cls.HEAD is not None
      has_bomb = cls.BOMB is not None
      has_tail = cls.TAIL is not None
      if has_head and has_bomb:
        return (cls.HEAD <= vals) & (vals < cls.BOMB)
      if has_head and has_tail:
        return (cls.HEAD <= vals) & (vals <= cls.TAIL)
      if has_head:
        return cls.HEAD <= vals
      if has_bomb:
        return vals < cls.BOMB
      if has_tail:
        return vals <= cls.TAIL
      return ~vals.isnull()
    return pd.Series(np.zeros(len(vals))).astype(np.bool)

  @classmethod
  def _bucket_scalar(cls, val, step, head):
    if pdutils.is_empty(val):
      return None
    if (cls.TAIL is not None) and (val == cls.TAIL):
      val = cls.TAIL - (0.5 * step)
    index = math.floor((val - head) / step)
    return head + (index * step)

  @classmethod
  def _bucket_pd(cls, val, step, head):
    if (cls.TAIL is not None):
      val = val.replace(to_replace=cls.TAIL, value=(cls.TAIL - 0.5 * step))
    index = np.floor((val - head) / step)
    return head + (index * step)

  @classmethod
  def _label_scalar(cls, bucket):
    if pdutils.is_empty(bucket):
      return None
    return cls.LABEL_FMT % (bucket,)

  @classmethod
  def _label_pd(cls, bucket):
    buckets = cls._to_series(bucket)
    labels = buckets.apply(cls._label_scalar)
    if len(labels) == 0:
      # Empty Series retains its original dtype after apply.
      labels = labels.astype(str)
    return labels

  @classmethod
  def is_valid(cls, val):
    if is_scalar(val):
      return cls._is_valid_scalar(val)
    return cls._is_valid_pd(val)

  @classmethod
  def coax(cls, obj, **kwargs):
    if is_scalar(obj):
      return cls._coax_scalar(obj, **kwargs)
    return cls._coax_pd(obj, **kwargs)

  @classmethod
  def bucket(cls, val, step, head=None):
    head = pdutils.coalesce(head, cls.HEAD, 0)
    if is_scalar(val):
      return cls._bucket_scalar(val, step=step, head=head)
    return cls._bucket_pd(val, step=step, head=head)

  @classmethod
  def label(cls, bucket):
    if is_scalar(bucket):
      return cls._label_scalar(bucket)
    return cls._label_pd(bucket)

  @classmethod
  def depth_step(cls, depth, head=None, bomb=None):
    head = pdutils.coalesce(head, cls.HEAD)
    bomb = pdutils.coalesce(bomb, cls.BOMB, cls.TAIL)
    return (bomb - head) / 2**depth

  @classmethod
  def labeler(cls, step, head=None, **kwargs):
    return GTypeLabeler(gtype=cls, step=step, head=head, **kwargs)


class Point(SuperGType):
  class _Coordinate(GTypeNumeric):
    LABEL_FMT = '%+011.04f'

    @classmethod
    def _coax_scalar(cls, obj):
      if pdutils.is_empty(obj):
        return None
      if not isinstance(obj, float):
        obj = float(obj)
      return obj if cls.is_valid(obj) else None

    @classmethod
    def _coax_pd(cls, obj):
      objs = cls._to_series(obj)
      objs = to_numeric(vals=objs)
      objs[~cls._is_valid_pd(objs)] = np.nan
      return objs

    @classmethod
    def labeler(cls, step=None, head=None, depth=None):
      if ((depth is not None and step is not None) or (depth is None and step is None)):
        raise ValueError('must provide one of depth, step/head: depth=%r, step=%r' % (depth, step))
      step = pdutils.coalesce(step, (depth is not None) and cls.depth_step(depth=depth))
      head = pdutils.coalesce(head, cls.HEAD)
      return super().labeler(step=step, head=head)

  class Latitude(_Coordinate):
    HEAD = -90.0
    BOMB = None
    TAIL = 90.0
    SANE_STEP = GTypeNumeric.depth_step(depth=10, head=HEAD, bomb=TAIL)

  class Longitude(_Coordinate):
    HEAD = -180.0
    BOMB = None
    TAIL = 180.0
    SANE_STEP = GTypeNumeric.depth_step(depth=10, head=HEAD, bomb=TAIL)


class Datetime(SuperGType):
  class Year(GTypeNumeric):
    LABEL_FMT = '%04d'

    @classmethod
    def _coax_scalar(cls, obj, fmt='%Y'):
      if pdutils.is_empty(obj):
        return None
      if isinstance(obj, (datetime.datetime, pd.Timestamp)):
        return obj.year
      if isinstance(obj, str) and (fmt is not None):
        return pd.to_datetime(obj, format=fmt, errors=COERCE).year
      return int(obj)

    @classmethod
    def _coax_pd(cls, obj, fmt='%Y'):
      objs = cls._to_series(obj)
      if pd_types.is_numeric_dtype(objs):
        return np.floor(objs)
      if fmt is not None:
        return pd.to_datetime(objs, format=fmt, errors=COERCE).dt.year
      return to_numeric(objs)

  class Month(GTypeNumeric):
    HEAD = 1
    BOMB = 13
    LABEL_FMT = '%02d'

    @classmethod
    def _coax_scalar(cls, obj, fmt='%m'):
      if pdutils.is_empty(obj):
        return None
      if isinstance(obj, (datetime.datetime, pd.Timestamp)):
        return obj.month
      if isinstance(obj, str) and (fmt is not None):
        return pd.to_datetime(obj, format=fmt, errors=COERCE).month
      return int(obj)

    @classmethod
    def _coax_pd(cls, obj, fmt='%m'):
      objs = cls._to_series(obj)
      if pd_types.is_numeric_dtype(objs):
        return np.floor(objs)
      if fmt is not None:
        return pd.to_datetime(objs, format=fmt, errors=COERCE).dt.month
      return to_numeric(objs)
