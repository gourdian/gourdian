import collections
import datetime
import itertools
import more_itertools
import numpy as np
import re
import sys

from gourdian import api
from gourdian.api import fetch as api_fetch
from gourdian.lib import errors as lib_errors
from gourdian.lib import gtypes
from gourdian.lib.sequences import Sequence, MultiSequence
from gourdian.utils import pdutils
from gourdian.utils import strutils


ENDPOINT_RE = '[a-z0-9][a-z0-9_]{2,23}'

USER = 'user'
DATASET = 'dataset'
TABLE = 'table'
LAYOUT = 'layout'
UPLOAD = 'upload'


def parse_endpointer(endpointer):
  """Returns Endpointer by parsing the provided endpointer (str, Endpointer supported).

  >>> parse_endpointer('user/dataset')
  <Endpointer 'user/dataset'>
  >>> parse_endpointer(Endpointer(user_name='user', dataset_name='dataset')
  <Endpointer 'user/dataset'>
  """
  if isinstance(endpointer, Endpointer):
    return endpointer
  return Endpointer.from_str(endpointer_str=endpointer)


def join_endpointer(user_name=None, dataset_name=None, table_name=None, layout_name=None):
  return Endpointer(user_name=user_name, dataset_name=dataset_name, table_name=table_name,
                    layout_name=layout_name)


class Endpointer:
  _ENDPOINTER_RE = re.compile(r'^(?:(\w+)(?:/(\w+)?(?:[.](\w+)?(?:@(\w+)?)?)?)?)?$')   # lol

  @classmethod
  def from_str(cls, endpointer_str):
    if not isinstance(endpointer_str, str):
      raise TypeError('cannot parse non-str: %r' % (endpointer_str,))
    match = cls._ENDPOINTER_RE.match(endpointer_str)
    if not match:
      raise ValueError('cannot parse: %r' % (endpointer_str,))
    user_name, dataset_name, table_name, layout_name = match.groups()
    return cls(user_name=user_name, dataset_name=dataset_name, table_name=table_name,
               layout_name=layout_name)

  def __init__(self, user_name=None, dataset_name=None, table_name=None, layout_name=None,
               upload_hash=None):
    if dataset_name and not user_name:
      raise ValueError('dataset_name requires user_name: user_name=%r' % (user_name,))
    if table_name and not dataset_name:
      raise ValueError('table_name requires dataset_name: dataset_name=%r' % (dataset_name,))
    if layout_name and not table_name:
      raise ValueError('layout_name requires table_name: table_name=%r' % (table_name,))
    if upload_hash and not table_name:
      raise ValueError('upload_hash requires table_name: table_name=%r' % (table_name,))
    if upload_hash and layout_name:
      raise ValueError('upload_hash must not have layout_name: layout_name=%r' % (layout_name,))
    self._user_name = user_name
    self._dataset_name = dataset_name
    self._table_name = table_name
    self._layout_name = layout_name
    self._upload_hash = upload_hash

  def __repr__(self):
    return '<%s %r>' % (self.__class__.__name__, str(self))

  def __str__(self):
    if self.upload_hash:
      fmt = '{user}/{dataset}.{table}${upload}'
    elif self.layout_name:
      fmt = '{user}/{dataset}.{table}@{layout}'
    elif self.table_name:
      fmt = '{user}/{dataset}.{table}'
    elif self.dataset_name:
      fmt = '{user}/{dataset}'
    elif self.user_name:
      fmt = '{user}'
    else:
      fmt = ''
    return fmt.format(user=self.user_name, dataset=self.dataset_name, table=self.table_name,
                      layout=self.layout_name, upload=self.upload_hash)

  def __eq__(self, obj):
    if isinstance(obj, str):
      return str(self) == obj
    return ((type(self) == type(obj))
            and (self.endpoint_type == obj.endpoint_type)
            and (self.user_name == obj.user_name)
            and (self.dataset_name == obj.dataset_name)
            and (self.table_name == obj.table_name)
            and (self.layout_name == obj.layout_name)
            and (self.upload_hash == obj.upload_hash))

  def __hash__(self):
    return hash(str(self))

  def __iter__(self):
    return itertools.chain((self.user_name, self.dataset_name, self.table_name, self.layout_name))

  @property
  def endpoint_type(self):
    if self.upload_hash:
      return UPLOAD
    if self.layout_name:
      return LAYOUT
    if self.table_name:
      return TABLE
    if self.dataset_name:
      return DATASET
    if self.user_name:
      return USER
    return None

  @property
  def user_name(self):
    return self._user_name

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def table_name(self):
    return self._table_name

  @property
  def layout_name(self):
    return self._layout_name

  @property
  def upload_hash(self):
    return self._upload_hash

  @property
  def user_endpointer(self):
    if self.user_name is None:
      return None
    return Endpointer(user_name=self.user_name)

  @property
  def dataset_endpointer(self):
    if self.dataset_name is None:
      return None
    return Endpointer(user_name=self.user_name, dataset_name=self.dataset_name)

  @property
  def table_endpointer(self):
    if self.table_name is None:
      return None
    return Endpointer(user_name=self.user_name, dataset_name=self.dataset_name,
                      table_name=self.table_name)

  @property
  def layout_endpointer(self):
    if self.layout_name is None:
      return None
    return Endpointer(user_name=self.user_name, dataset_name=self.dataset_name,
                      table_name=self.table_name, layout_name=self.layout_name)

  @property
  def upload_endpointer(self):
    if self.upload_hash is None:
      return None
    return Endpointer(user_name=self.user_name, dataset_name=self.dataset_name,
                      table_name=self.table_name, upload_hash=self.upload_hash)

  def to_dict(self):
    return dict(
      user_name=self.user_name,
      dataset_name=self.dataset_name,
      table_name=self.table_name,
      layout_name=self.layout_name,
      upload_hash=self.upload_hash,
    )


class Manifest:
  @classmethod
  def from_js(cls, manifest_js, client=None):
    client = api.DEFAULT_CLIENT if client is None else client   # NOTE: avoids circular import
    # 1. Manifest.
    generated_at = datetime.datetime.fromisoformat(manifest_js['generated_at'])
    manifest = cls(generated_at=generated_at, client=client)
    # 2. User.
    user_js = manifest_js['user']
    user = User.from_js(user_js=user_js, client=client)
    # 3. Dataset.
    dataset_js = manifest_js['dataset']
    dataset = Dataset.from_js(user=user, dataset_js=dataset_js)
    manifest.set_dataset(dataset=dataset)
    return manifest

  def __init__(self, generated_at, client):
    self._generated_at = generated_at
    self._client = client
    # Placeholders.
    self._dataset = None

  def __repr__(self):
    return '<%s %r at=%r>' % (self.__class__.__name__, str(self.endpointer),
                              self.generated_at.isoformat())

  def set_dataset(self, dataset):
    if self._dataset is not None:
      raise lib_errors.AlreadyBoundError('dataset already set: %s' % (self._dataset,))
    self._dataset = dataset.set_parent(manifest=self)
    return dataset

  @property
  def client(self):
    return self._client

  @property
  def dataset(self):
    return self._dataset

  @property
  def user(self):
    return self.dataset.user

  @property
  def generated_at(self):
    return self._generated_at

  @property
  def endpointer(self):
    return self.dataset.endpointer


class Endpoint:
  def __eq__(self, obj):
    return (type(obj) == type(self)) and (self.endpointer == obj.endpointer)

  def __hash__(self):
    return hash(self.endpointer)

  @property
  def endpointer(self):
    raise NotImplementedError()

  @property
  def endpoint_type(self):
    return self.endpointer.endpoint_type


class User(Endpoint):
  @classmethod
  def from_js(cls, user_js, client=None):
    return cls(name=user_js['name'], client=client)

  def __init__(self, name, client=None):
    self._name = name
    self._client = client

  def __repr__(self):
    return '<%s %r>' % (self.__class__.__name__, str(self.endpointer))

  def __str__(self):
    return str(self.endpointer)

  @property
  def name(self):
    return self._name

  @property
  def endpointer(self):
    return Endpointer(user_name=self.name)


class Dataset(Endpoint):
  @classmethod
  def from_js(cls, user, dataset_js):
    dataset = cls(
      user=user,
      name=dataset_js['name'],
      display_name=dataset_js['display_name'],
      description=dataset_js['description'],
      short_description=dataset_js['short_description'],
      homepage_url=dataset_js['homepage_url'],
      download_url=dataset_js['download_url'],
      license_text=dataset_js['license_text'],
      license_url=dataset_js['license_url'],
    )
    tables = [Table.from_js(table_js=js) for js in dataset_js['tables']]
    for table in tables:
      dataset.add_table(table=table)
    return dataset

  def __init__(self, user, name, display_name=None, description=None, short_description=None,
               homepage_url=None, download_url=None, license_text=None, license_url=None):
    super().__init__()
    self._user = user
    self._name = name
    self._display_name = display_name
    self._description = description
    self._short_description = short_description
    self._homepage_url = homepage_url
    self._download_url = download_url
    self._license_text = license_text
    self._license_url = license_url
    # Placeholders.
    self._manifest = None
    self._tables = {}

  def __repr__(self):
    return ('<%s %r tables=%r>' %
            (self.__class__.__name__, str(self.endpointer), [t.name for t in self.tables]))

  def __str__(self):
    return str(self.endpointer)

  def describe(self, f=sys.stdout):
    has_license = self.license_url or self.license_text
    if self.manifest.generated_at:
      generated_at_str = self.manifest.generated_at.isoformat()
    else:
      generated_at_str = None
    # Format available sources.
    tables_parts = []
    for table in self.tables:
      num_rows_str = strutils.format_number(table.layouts[0].num_rows())
      tables_parts.append('- %s (%s rows)' % (table.name, num_rows_str))
    tables_str = '\n'.join(tables_parts)
    parts = [
      '# %s' % (self.display_name,),
      'Endpointer: `%s`' % (self.endpointer,),
      '',
      self.description if self.description else None,
      '' if self.description else None,
      'Homepage: %s' % (self.homepage_url,) if self.homepage_url else None,
      'Downloaded From: %s' % (self.download_url,) if self.download_url else None,
      'Generated At: %s' % (generated_at_str,) if generated_at_str else None,
      '' if (self.homepage_url or self.download_url or generated_at_str) else None,
      '## Available Tables (%d)' % (len(self.tables),),
      tables_str,
      '',
      'License Summary' if has_license else None,
      '---------------' if has_license else None,
      'Full License: %s' % (self.license_url,) if self.license_url else None,
      '' if self.license_url and self.license_text else None,
      self.license_text if self.license_text else None,
    ]
    ret = '\n'.join(p for p in parts if p is not None)
    if not f:
      return f
    f.write(ret)
    f.flush()

  def set_parent(self, manifest):
    if self._manifest is not None:
      raise lib_errors.AlreadyBoundError('dataset is already bound: %r' % (self._manfiest,))
    self._manifest = manifest
    return self

  @property
  def client(self):
    return self.manifest.client

  @property
  def manifest(self):
    return self._manifest

  def add_table(self, table):
    if table.name in self._tables:
      raise lib_errors.DuplicateNameError('dataset already has table with name: %s' % (table.name,))
    table.set_parent(dataset=self)
    self._tables[table.name] = table
    return table

  @property
  def tables(self):
    return tuple(sorted(self._tables.values(), key=lambda x: x.name))

  def table(self, name):
    return self._tables[name]

  @property
  def name(self):
    return self._name

  @property
  def display_name(self):
    return self._display_name

  @property
  def description(self):
    return self._description

  @property
  def short_description(self):
    return self._short_description

  @property
  def homepage_url(self):
    return self._homepage_url

  @property
  def download_url(self):
    return self._download_url

  @property
  def license_url(self):
    return self._license_url

  @property
  def license_text(self):
    return self._license_text

  @property
  def user(self):
    return self._user

  @property
  def endpointer(self):
    return Endpointer(user_name=self.user.name, dataset_name=self.name)


class Table(Endpoint):
  @classmethod
  def from_js(cls, table_js):
    table = cls(
      name=table_js['name'],
      display_name=table_js['display_name'],
      description=table_js['description'],
      short_description=table_js['short_description'],
      homepage_url=table_js['homepage_url'],
      download_url=table_js['download_url'],
      columns=table_js['columns'],
    )
    layouts = [Layout.from_js(layout_js=js) for js in table_js['layouts']]
    for layout in layouts:
      table.add_layout(layout=layout)
    return table

  def __init__(self, name, columns=(), display_name=None, description=None, short_description=None,
               homepage_url=None, download_url=None):
    super().__init__()
    self._name = name
    self._columns = tuple(columns)
    self._display_name = display_name
    self._description = description
    self._short_description = short_description
    self._homepage_url = homepage_url
    self._download_url = download_url
    # Placeholders.
    self._dataset = None
    self._layouts = {}

  def __repr__(self):
    return ('<%s %r layouts=%r>' %
            (self.__class__.__name__, str(self.endpointer), [x.name for x in self.layouts]))

  def __str__(self):
    return str(self.endpointer)

  def describe(self, f=sys.stdout):
    columns_str = '\n'.join('- %s' % (c,) for c in self.columns)
    layout_parts = []
    for layout in self.layouts:
      layout_parts.append('- %s' % (layout.name,))
      for label_column in sorted(layout.label_columns, key=lambda x: x.name):
        head = label_column.sequence.head
        bomb = label_column.sequence.bomb
        layout_parts.append('  + %s' % (str(label_column.labeler()),))
        layout_parts.append('    - [%r ... %r] (len=%r)' % (head, bomb, bomb-head))
    layouts_str = '\n'.join(layout_parts)
    parts = [
      '# %s' % (self.display_name,),
      'Endpointer: `%s`' % (self.endpointer,),
      '',
      self.description,
      '' if self.description else None,
      'Homepage: %s' % (self.homepage_url,) if self.homepage_url else None,
      'Downloaded From: %s' % (self.download_url,) if self.download_url else None,
      '' if self.homepage_url or self.download_url else None,
      '## Columns (%d)' % (len(self.columns),),
      columns_str,
      '',
      '## Layouts (%d)' % (len(self.layouts),),
      layouts_str,
    ]
    ret = '\n'.join(p for p in parts if p is not None)
    if not f:
      return ret
    f.write(ret)
    f.flush()

  def set_parent(self, dataset):
    if self._dataset is not None:
      raise lib_errors.AlreadyBoundError('table is already bound: %r' % (self._dataset,))
    self._dataset = dataset
    return self

  @property
  def client(self):
    return self.dataset.client

  @property
  def dataset(self):
    return self._dataset

  def add_layout(self, layout):
    if layout.name in self._layouts:
      raise lib_errors.DuplicateNameError('table already has layout with name: %s' % (layout.name,))
    layout.set_parent(table=self)
    self._layouts[layout.name] = layout
    return layout

  @property
  def layouts(self):
    return tuple(sorted(self._layouts.values(), key=lambda x: x.name))

  def layout(self, name):
    return self._layouts[name]

  @property
  def name(self):
    return self._name

  @property
  def columns(self):
    return self._columns

  @property
  def display_name(self):
    return self._display_name

  @property
  def description(self):
    return self._description

  @property
  def short_description(self):
    return self._short_description

  @property
  def homepage_url(self):
    return self._homepage_url

  @property
  def download_url(self):
    return self._download_url

  @property
  def user(self):
    return self.dataset.user

  @property
  def manifest(self):
    return self.dataset.manifest

  @property
  def endpointer(self):
    return Endpointer(user_name=self.user.name, dataset_name=self.dataset.name,
                      table_name=self.name)


class Layout(Endpoint):
  @classmethod
  def from_js(cls, layout_js):
    sequence = MultiSequence(sequences=[Sequence(**seq) for seq in layout_js['sequence']])
    layout = cls(
      name=layout_js['name'],
      sequence=sequence,
    )
    label_columns = [LabelColumn.from_js(sequence=sequence.sequences[i], label_column_js=js)
                     for i, js in enumerate(layout_js['label_columns'])]
    for label_column in label_columns:
      layout.add_label_column(label_column=label_column)
    arrays = [LayoutArray.from_js(name=k, array_js=js) for k, js in layout_js['arrays'].items()]
    for array in arrays:
      layout.add_array(array=array)
    return layout

  def __init__(self, name, sequence):
    super().__init__()
    self._name = name
    self._sequence = sequence
    # Placeholders.
    self._table = None
    self._label_columns = collections.OrderedDict()
    self._arrays = {}

  def __repr__(self):
    return ('<%s %r label_columns=%r>' %
            (self.__class__.__name__, str(self.endpointer), [x.name for x in self.label_columns]))

  def __str__(self):
    return str(self.endpointer)

  def set_parent(self, table):
    if self._table is not None:
      raise lib_errors.AlreadyBoundError('layout is already bound: %r' % (self._table,))
    self._table = table
    return self

  @property
  def client(self):
    return self.table.client

  @property
  def table(self):
    return self._table

  def add_label_column(self, label_column):
    if label_column.name in self._label_columns:
      raise lib_errors.DuplicateNameError('layout already has label_column with name: %s' %
                                      (label_column.name,))
    column_index = len(self._label_columns)
    label_column.set_parent(layout=self, column_index=column_index)
    self._label_columns[label_column.name] = label_column
    return label_column

  @property
  def label_columns(self):
    return tuple(self._label_columns.values())

  def label_column(self, name=None, gtype=None, label_kwargs=None):
    # Create a set of candidate columns that match the search criteria.
    candidates = list(self.label_columns)
    if name is not None:
      candidates = [x for x in candidates if x.name == name]
    if gtype is not None:
      candidates = [x for x in candidates if x.gtype == gtype]
    if label_kwargs is not None:
      candidates = [x for x in candidates if x.labeler().kwargs == label_kwargs]
    # Assert that we found exactly 1 candidate, and return it.
    if len(candidates) == 0:
      target = {k: v for k, v in [('name', name), ('gtype', gtype), ('label_kwargs', label_kwargs)]
                if not pdutils.is_empty(v)}
      raise lib_errors.NoLabelColumnsError('0 columns found on %r for %r' %
                                           (str(self.endpointer), target,))
    if len(candidates) > 1:
      target = {k: v for k, v in [('name', name), ('gtype', gtype), ('label_kwargs', label_kwargs)]
                if not pdutils.is_empty(v)}
      raise lib_errors.MultipleLabelColumnsError('%d columns found on %r for %r: %r' %
                                                 (str(self.endpointer), target, len(candidates),
                                                  candidates))
    return more_itertools.one(candidates)

  def add_array(self, array):
    if array.name in self._arrays:
      raise lib_errors.DuplicateNameError('layout already has array with name: %s' % (array.name,))
    array.set_parent(layout=self)
    self._arrays[array.name] = array
    return array

  @property
  def arrays(self):
    return tuple(sorted(self._arrays.values(), key=lambda x: x.name))

  def array(self, name):
    return self._arrays[name]

  def num_rows(self):
    return self.array(name='num_rows').sum

  def chunks(self):
    for bucket in self.sequence:
      yield self.chunk(bucket=bucket)

  def chunk(self, bucket):
    return LayoutChunk(layout=self, bucket=bucket)

  @property
  def name(self):
    return self._name

  @property
  def sequence(self):
    return self._sequence

  @property
  def dataset(self):
    return self.table.dataset

  @property
  def user(self):
    return self.dataset.user

  @property
  def manifest(self):
    return self.dataset.manifest

  @property
  def endpointer(self):
    return Endpointer(user_name=self.user.name, dataset_name=self.dataset.name,
                      table_name=self.table.name, layout_name=self.name)


class LabelColumn:
  @classmethod
  def from_js(cls, label_column_js, sequence):
    label_kwargs = {'head': sequence.head, 'step': float(sequence.step)}
    return cls(
      name=label_column_js['name'],
      gtype=gtypes.gtype(qualname=label_column_js['gtype']),
      **label_kwargs,
    )

  def __init__(self, name, gtype, **label_kwargs):
    super().__init__()
    self._name = name
    self._gtype = gtype
    self._label_kwargs = label_kwargs or {}
    # Placeholders.
    self._layout = None
    self._column_index = None

  def __repr__(self):
    label_str = ', '.join('%s=%r' % (k, v) for k, v in self._label_kwargs.items())
    return '<{cls} {name} of {endpointer} {gtype}{label_kwargs}>'.format(
      cls=self.__class__.__name__,
      name=repr(self.name),
      endpointer=repr(str(self.layout.endpointer)),
      gtype=str(self.gtype.qualname()) if self.gtype else None,
      label_kwargs=('(%s)' % (label_str,) if label_str else ''),
    )

  def __str__(self):
    return self.name

  def set_parent(self, layout, column_index):
    if self._layout is not None:
      raise lib_errors.AlreadyBoundError('label_column is already bound: %r' % (self._layout,))
    self._layout = layout
    self._column_index = column_index
    return self

  @property
  def client(self):
    return self.layout.client

  @property
  def layout(self):
    return self._layout

  @property
  def column_index(self):
    return self._column_index

  @property
  def sequence(self):
    return self.layout.sequence.sequences[self.column_index]

  @property
  def name(self):
    return self._name

  @property
  def gtype(self):
    return self._gtype

  @property
  def super_gtype(self):
    return self.gtype.super_gtype

  @property
  def label_kwargs(self):
    return dict(self._label_kwargs)

  def labeler(self):
    return self.gtype.labeler(**self._label_kwargs)


class LayoutArray:
  @classmethod
  def from_js(cls, name, array_js):
    return cls(
      name=name,
      sum=array_js['sum'],
      min=array_js['min'],
      max=array_js['max'],
      num_present=array_js['num_present'],
      num_empty=array_js['num_empty'],
    )

  @classmethod
  def read_array_values(cls, f):
    # NOTE: np.fromfile does not work with "files" returned by gzip.open().
    return np.frombuffer(f.read(), dtype='<u4')

  def __init__(self, name, sum, min, max, num_present, num_empty):
    super().__init__()
    self._name = name
    self._sum = sum
    self._min = min
    self._max = max
    self._num_present = num_present
    self._num_empty = num_empty
    # Placeholders.
    self._layout = None
    self._values = None

  def __repr__(self):
    return ('<%s %r sum=%d num_present=%d>' %
            (self.__class__.__name__, self.name, self.sum, self.num_present))

  def __len__(self):
    return self.num_present + self.num_empty

  def __iter__(self):
    return itertools.chain(self.values().flatten())

  def set_parent(self, layout):
    if self._layout is not None:
      raise lib_errors.AlreadyBoundError('array is already bound: %r' % (self._layout,))
    self._layout = layout
    return self

  @property
  def client(self):
    return self.layout.client

  @property
  def layout(self):
    return self._layout

  @property
  def name(self):
    return self._name

  @property
  def sum(self):
    return self._sum

  @property
  def min(self):
    return self._min

  @property
  def max(self):
    return self._max

  @property
  def num_present(self):
    return self._num_present

  @property
  def num_empty(self):
    return self._num_empty

  def values(self):
    if self._values is None:
      values = self.client.layout_array_values(
        endpointer=self.layout.endpointer,
        array_name=self.name,
      )
      assert len(values) == len(self)
      # Reshape values to fit the shape of the matching layout's sequence.
      self._values = np.reshape(values, self.layout.sequence.shape)
    return self._values


class LayoutChunk:
  def __init__(self, layout, bucket):
    self._layout = layout
    self._bucket = bucket
    # Placeholders.
    self._label = None
    self._filename = None
    # Checkrep.
    if len(bucket) != len(layout.label_columns):
      raise ValueError('bucket mismatch with layout.label_columns(len=%d): %r' %
                       (len(layout.label_columns), bucket))

  def __repr__(self):
    return '<%s %r of %r>' % (self.__class__.__name__, self.bucket, str(self.layout))

  def __str__(self):
    return self.filename

  @property
  def client(self):
    return self.layout.client

  @property
  def layout(self):
    return self._layout

  @property
  def bucket(self):
    return self._bucket

  @property
  def label(self):
    if self._label is None:
      label_columns = self.layout.label_columns
      self._label = tuple(c.labeler().label(b) for (b, c) in zip(self.bucket, label_columns))
    return self._label

  @property
  def indices(self):
    return self.layout.sequence.left_indices_of(values=self.bucket)

  @property
  def filename(self):
    if self._filename is None:
      self._filename = api_fetch.layout_chunk_filename(label=self.label)
    return self._filename

  def num_rows(self):
    return self.layout.array(name='num_rows').values()[self.indices]

  def df(self):
    return self.client.chunk_df(
      endpointer=self.layout.endpointer,
      filename=self.filename,
    )

  def dfs(self, page_size=1_000):
    return self.client.chunk_dfs(
      endpointer=self.layout.endpointer,
      filename=self.filename,
      page_size=page_size,
    )
