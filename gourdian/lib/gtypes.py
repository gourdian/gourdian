import collections
import collections.abc
import datetime
import decimal
import math
import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import re
import string

from gourdian.utils import pdutils


COERCE = 'coerce'

ROUND_UP_CONTEXT = decimal.Context(rounding=decimal.ROUND_HALF_UP)

GTYPE_RE = r'[a-zA-Z0-9][a-zA-Z0-9_]{2,23}'
SUPER_GTYPE_NAME = 'super_gtype_name'
GTYPE_NAME = 'gtype_name'
GTYPE_ID = 'gtype_id'


def is_scalar(val):
  if isinstance(val, str):
    return True
  return not isinstance(val, collections.abc.Iterable)


def to_numeric(vals):
  if len(vals) > 0:
    # Work-around for https://github.com/modin-project/modin/issues/2833
    return pd.to_numeric(vals, errors=COERCE)
  return vals


class Qualname:
  _QUALNAME_PAT = r'^({sgn})(?:[.]({gtn})(?:[#]({gid}))?)?$'
  _QUALNAME_RE = re.compile(_QUALNAME_PAT.format(sgn=GTYPE_RE, gtn=GTYPE_RE, gid=r'[^#]+'))

  @classmethod
  def from_str(cls, qualname_str):
    if isinstance(qualname_str, cls):
      return qualname_str
    if not isinstance(qualname_str, str):
      raise TypeError('cannot parse non-str: %r' % (qualname_str,))
    match = cls._QUALNAME_RE.match(qualname_str)
    if not match:
      raise ValueError('cannot parse: %r' % (qualname_str,))
    super_gtype_name, gtype_name, gtype_id = match.groups()
    return cls(super_gtype_name=super_gtype_name, gtype_name=gtype_name, gtype_id=gtype_id)

  def __init__(self, super_gtype_name, gtype_name, gtype_id):
    self._super_gtype_name = super_gtype_name
    self._gtype_name = gtype_name
    self._gtype_id = gtype_id

  def __repr__(self):
    return '<%s %r>' % (self.__class__.__name__, str(self))

  def __str__(self):
    if self.gtype_id:
      fmt = '{sgn}.{gtn}#{gid}'
    elif self.gtype_name:
      fmt = '{sgn}.{gtn}'
    elif self.super_gtype_name:
      fmt = '{sgn}'
    else:
      fmt = ''
    return fmt.format(sgn=self.super_gtype_name, gtn=self.gtype_name, gid=self.gtype_id)

  def __eq__(self, obj):
    if isinstance(obj, str):
      return str(self) == obj
    return ((type(self) == type(obj))
            and (self.qualname_type == obj.qualname_type)
            and (self.super_gtype_name == obj.super_gtype_name)
            and (self.gtype_name == obj.gtype_name)
            and (self.gtype_id == obj.gtype_id))

  def __hash__(self):
    return hash(str(self))

  def __iter__(self):
    return itertools.chain((self.super_gtype_name, self.gtype_name, self.gtype_id))

  @property
  def qualname_type(self):
    if self.gtype_id:
      return GTYPE_ID
    if self.gtype_name:
      return GTYPE_NAME
    if self.super_gtype_name:
      return SUPER_GTYPE_NAME
    return None

  @property
  def super_gtype_name(self):
    return self._super_gtype_name

  @property
  def gtype_name(self):
    return self._gtype_name

  @property
  def gtype_id(self):
    return self._gtype_id

  @property
  def basename(self):
    if self.gtype_name is None:
      return None
    return Qualname(
      super_gtype_name=self.super_gtype_name,
      gtype_name=self.gtype_name,
      gtype_id=None,
    )


def super_gtype(name):
  """Returns the SuperGType class with a given name."""
  return GTypeMeta.super_gtype(name=name)


def gtype(qualname, create_kwargs=None):
  """Returns the GType class with a given qualname; e.g. <SuperGType.name>.<GType.name>.

  If create_kwargs are provided, the requested gtype's create() method is called with those args.
  """
  qualname = Qualname.from_str(qualname_str=str(qualname))
  gtype = GTypeMeta.base_gtype(qualname=qualname)
  gtype_id = qualname.gtype_id
  return gtype.create(gtype_id=gtype_id, **create_kwargs) if create_kwargs is not None else gtype


def all_gtypes():
  """Returns all defined GType classes."""
  return GTypeMeta.all_gtypes()


def is_gtype(value):
  return isinstance(value, type) and issubclass(value, GType)


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
    return ('<%s %s(%s)>' % (self.__class__.__name__, self._gtype.qualname(), kw_str))

  def __str__(self):
    return self.name

  @property
  def name(self):
    if self._name is None:
      kw = ((k, v) for k, v in self._kwargs.items() if not pdutils.is_empty(v))
      kw_str = ', '.join('%s=%r' % (k, v) for k, v in kw)
      self._name = '%s(%s)' % (self._gtype.qualname(), kw_str)
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
      # TODO(rob): Does assigning klass.qualname conflict with GType.qualname()...
      # klass.qualname = klass.__qualname__
      qualname = klass.qualname()
      klass.name = qualname.gtype_name
      cls._GTYPES[str(qualname)] = klass
      # If the super_gtype has already been created, associate it now.
      super_gtype_name = str(qualname).split('.', 1)[0]
      if super_gtype_name in cls._SUPER_GTYPES:
        super_gtype = cls.super_gtype(name=super_gtype_name)
        klass.super_gtype = super_gtype
        super_gtype.gtypes = cls.gtypes_of(super_gtype=super_gtype)
    else:
      raise TypeError('cannot register type: %r' % (name,))
    return klass

  def __repr__(cls):
    if issubclass(cls, GType):
      return '<gtype %s>' % (cls.qualname(),)
    return '<super_gtype %s>' % (cls.name,)

  def __str__(cls):
    if issubclass(cls, GType):
      return str(cls.qualname())
    return cls.name

  @classmethod
  def super_gtype(cls, name):
    return cls._SUPER_GTYPES[name]

  @classmethod
  def base_gtype(cls, qualname):
    qualname = Qualname.from_str(qualname_str=str(qualname))
    basename = str(qualname.basename)
    return cls._GTYPES[basename]

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
  def qualname(cls):
    return Qualname.from_str(qualname_str=cls.__name__)

  @classmethod
  def base_gtype(cls, name):
    qualname = '{}.{}'.format(cls.__name__, name)
    return GTypeMeta.base_gtype(qualname)

  def __str__(self):
    return self.name


class GType(metaclass=GTypeMeta):
  create_kwargs = None

  @classmethod
  def qualname(cls):
    qualname_str = cls.__qualname__
    return Qualname.from_str(qualname_str=qualname_str)

  @classmethod
  def _to_series(cls, val):
    if isinstance(val, pd.Series):
      series = val
    elif isinstance(val, pd.DataFrame):
      raise ValueError("cannot convert DataFrame to Series; try df['col_name']")
    else:
      series = pd.Series(val)
    return series.rename(series.name or str(cls.qualname()))

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
  def labeler(cls, step, head=None, **kwargs):
    return GTypeLabeler(gtype=cls, step=step, head=head, **kwargs)


class GTypeNumericBucket(GType):
  """A GType for types that have numeric bucket values."""
  # BOMB/TAIL are mutually exclusive: valid if val < BOMB or val <= TAIL.
  HEAD = None
  BOMB = None
  TAIL = None
  # Format string to use when converting a numeric bucket into a string label.
  LABEL_FMT = '{:0.04f}'
  # Step size to use when building a Query object with no pre-defined step info.
  SANE_STEP = 1

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
  def _label_scalar_nocontext(cls, bucket):
    if pdutils.is_empty(bucket):
      return None
    return cls.LABEL_FMT.format(decimal.Decimal(bucket))

  @classmethod
  def _label_scalar(cls, bucket):
    context = decimal.getcontext()
    try:
      decimal.setcontext(ROUND_UP_CONTEXT)
      return cls._label_scalar_nocontext(bucket=bucket)
    finally:
      decimal.setcontext(context)

  @classmethod
  def _label_pd(cls, bucket):
    context = decimal.getcontext()
    try:
      decimal.setcontext(ROUND_UP_CONTEXT)
      buckets = cls._to_series(bucket)
      labels = buckets.apply(cls._label_scalar)
      if len(labels) == 0:
        # Empty Series retains its original dtype after apply.
        labels = labels.astype(str)
      return labels
    finally:
      decimal.setcontext(context)

  @classmethod
  def depth_step(cls, depth, head=None, bomb=None):
    head = pdutils.coalesce(head, cls.HEAD)
    bomb = pdutils.coalesce(bomb, cls.BOMB, cls.TAIL)
    return (bomb - head) / 2**depth


class GTypeNumeric(GTypeNumericBucket):
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
    return pd.Series(np.zeros(len(vals), dtype=np.bool))

  @classmethod
  def _coax_scalar(cls, obj):
    if pdutils.is_empty(obj):
      return None
    if not isinstance(obj, float):
      obj = float(obj)
    return obj if cls._is_valid_scalar(obj) else None

  @classmethod
  def _coax_pd(cls, obj):
    obj = cls._to_series(obj)
    obj = to_numeric(vals=obj)
    obj[~cls._is_valid_pd(obj)] = np.nan
    return obj


class GTypeEnum(GTypeNumericBucket):
  """Abstract GType representing string values that define an enum of finite values.

  This class cannot be used on its own, as it has an empty set of valid values.  Instead, an enum's
  valid values should be provided to create() which returns a new subclass configured to use them.
  """
  create_kwargs = {'values': []}
  VALUES = {}

  HEAD = 0
  BOMB = 0
  TAIL = None

  LABEL_FMT = '{:d}'

  @classmethod
  def create(cls, gtype_id, values):
    values = dict(values) if isinstance(values, dict) else {v: i for i, v in enumerate(values)}
    name = 'Enum#%s' % (gtype_id,)
    return type(name, (cls,), {
      '__qualname__': 'String.%s' % (name,),
      'create_kwargs': {'values': values},
      'VALUES': values,
      'BOMB': len(values),
      'LABEL_FMT': '{:0%dd}' % (len(str(len(values))),),
    })

  @classmethod
  def _is_valid_scalar(cls, val):
    return val in cls.VALUES

  @classmethod
  def _is_valid_pd(cls, val):
    val = cls._to_series(val)
    return val.isin(set(cls.VALUES))

  @classmethod
  def _coax_scalar(cls, obj):
    if pdutils.is_empty(obj):
      return None
    obj = str(obj)
    obj = obj.strip()
    return obj if cls._is_valid_scalar(obj) else None

  @classmethod
  def _coax_pd(cls, obj):
    obj = cls._to_series(obj)
    obj = obj.str.strip()
    obj[~cls._is_valid_pd(obj)] = np.nan
    return obj

  @classmethod
  def _val_to_enum(cls, val):
    if pdutils.is_empty(val):
      return None
    return cls.VALUES[val]

  @classmethod
  def _bucket_scalar(cls, val, step, head=None):
    val = cls._val_to_enum(val=val)
    return GTypeNumericBucket._bucket_scalar(val=val, step=step, head=head)

  @classmethod
  def _bucket_pd(cls, val, step, head=None):
    val = cls._to_series(val)
    val = val.apply(func=cls._val_to_enum)
    return GTypeNumericBucket._bucket_pd(val=val, step=step, head=head)

  @classmethod
  def _label_scalar_nocontext(cls, bucket):
    if pdutils.is_empty(bucket):
      return None
    return cls.LABEL_FMT.format(int(bucket))


class GTypeSlug(GTypeNumericBucket):
  """GType representing string values that can be meaningfully converted to slugs."""
  # There are 37 valid slug characters: [0-9] + [a-z] + [_] = 10 + 26 + 1
  CLEANUP_RE = re.compile(r'[ ]+')
  SLUG_RE = re.compile(r'[^a-z0-9_]')
  SLUG_BASE = 37
  SLUG_ORDS = dict([(x, i) for i, x in enumerate(string.digits)]
                   + [(x, i+10) for i, x in enumerate(string.ascii_lowercase)]
                   + [('_', 36)])
  # Compute slug sequence value using the first 3 characters.
  SLUG_WIDTH = 3

  HEAD = 0
  BOMB = (SLUG_BASE**SLUG_WIDTH)
  TAIL = None

  LABEL_FMT = '{:0%dd}' % (len(str(BOMB)),)

  @classmethod
  def _is_valid_scalar(cls, val):
    return cls.SLUG_RE.search(val) is None

  @classmethod
  def _is_valid_pd(cls, val):
    val = cls._to_series(val)
    return ~val.str.contains(pat=cls.SLUG_RE)

  @classmethod
  def _coax_scalar(cls, obj):
    if pdutils.is_empty(obj):
      return None
    obj = str(obj)
    obj = obj.strip()
    obj = obj.lower()
    obj = cls.CLEANUP_RE.sub('_', obj)
    obj = cls.SLUG_RE.sub('', obj)
    return obj if cls._is_valid_scalar(obj) else None

  @classmethod
  def _coax_pd(cls, obj):
    obj = cls._to_series(obj)
    obj = obj.str.strip()
    obj = obj.str.lower()
    obj = obj.str.replace(cls.CLEANUP_RE, '_')
    obj = obj.str.replace(cls.SLUG_RE, '')
    obj[~cls._is_valid_pd(obj)] = np.nan
    return obj

  @classmethod
  def _snug_slug_to_float(cls, slug):
    if pdutils.is_empty(slug):
      return None
    ret = 0
    for i, x in enumerate(reversed(slug)):
      ret += (cls.SLUG_BASE**i) * cls.SLUG_ORDS[x]
    return ret

  @classmethod
  def _bucket_scalar(cls, val, step, head=None):
    # A. Convert the slug to a float in [0.0, 1.0).
    slug = (val + ('0' * cls.SLUG_WIDTH))[:cls.SLUG_WIDTH]
    val = cls._snug_slug_to_float(slug=slug)
    # B. Bucket that float using step/head.
    return GTypeNumericBucket._bucket_scalar(val=val, step=step, head=head)

  @classmethod
  def _bucket_pd(cls, val, step, head=None):
    # A. Convert the slug to a float in [0.0, 1.0).
    val = cls._to_series(val)
    val = val + ('0' * cls.SLUG_WIDTH)
    val = val.str.slice(start=0, stop=cls.SLUG_WIDTH)
    val = val.apply(func=cls._snug_slug_to_float)
    # B. Bucket that float using step/head.
    return GTypeNumericBucket._bucket_pd(val=val, step=step, head=head)

  @classmethod
  def _label_scalar_nocontext(cls, bucket):
    if pdutils.is_empty(bucket):
      return None
    return cls.LABEL_FMT.format(int(bucket))


class String(SuperGType):
  class Enum(GTypeEnum):
    pass

  class Slug(GTypeSlug):
    pass


class Point(SuperGType):
  class _Coordinate(GTypeNumeric):
    LABEL_FMT = '{:+09.4f}'

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
    LABEL_FMT = '{:04.0f}'

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
    LABEL_FMT = '{:02.0f}'

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
