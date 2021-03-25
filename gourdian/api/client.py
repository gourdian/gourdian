import gzip
import json
import pandas as pd
import urllib.parse

from gourdian.api import cache as api_cache
from gourdian.api import fetch as api_fetch
from gourdian.lib import lib
from gourdian.lib import queries


class ClientError(Exception):
  pass


class EndpointError(Exception):
  pass


class GourdianClient:
  def __init__(self, cache_factory=api_cache.LocalCache, url_root=api_fetch.DEFAULT_URL_ROOT):
    self.cache = cache_factory()
    self.url_root = url_root

  def __repr__(self):
    return '<%s url_root=%r cache=%r>' % (self.__class__.__name__, self.url_root, str(self.cache))

  def open(self, url, is_gz=None):
    cached = self.cache.get(url=url)
    if not cached:
      response = api_fetch.fetch(url=url)
      cached = self.cache.put(url, response)
    is_gz = urllib.parse.urlparse(url).path.endswith('.gz') if is_gz is None else is_gz
    if is_gz:
      return gzip.GzipFile(fileobj=cached, mode='rb')
    return cached

  def manifest(self, endpointer):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.DATASET:
      raise EndpointError('endpoint must be dataset: %s' % (endpointer,))
    url = api_fetch.manifest_url(
      user_name=endpointer.user_name,
      dataset_name=endpointer.dataset_name,
    )
    with self.open(url=url) as f:
      manifest_js = json.load(f)
      return lib.Manifest.from_js(manifest_js=manifest_js, client=self)

  def endpoint(self, endpointer, require_types=None):
    if isinstance(endpointer, lib.Endpoint):
      endpoint = endpointer
    else:
      endpointer = lib.parse_endpointer(endpointer=endpointer)
      endpoint_type = endpointer.endpoint_type
      if endpoint_type == lib.DATASET:
        endpoint = self.dataset(endpointer=endpointer)
      elif endpoint_type == lib.TABLE:
        endpoint = self.table(endpointer=endpointer)
      elif endpoint_type == lib.LAYOUT:
        endpoint = self.layout(endpointer=endpointer)
      else:
        raise TypeError('cannot extract endpoint of type %s: %r' % (endpoint_type, endpointer))
    if endpoint.client is not self:
      raise ClientError('endpoint was not created by this client: %r' % (endpoint.client,))
    if require_types is not None and endpoint.endpoint_type not in require_types:
      raise TypeError('endpoint must be in %r: %r' % (require_types, endpoint))
    return endpoint

  def dataset(self, endpointer):
    manifest = self.manifest(endpointer=endpointer)
    return manifest.dataset

  def table(self, endpointer):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.TABLE:
      raise EndpointError('endpoint must be table: %s' % (endpointer,))
    dataset_endpointer = endpointer.dataset_endpointer
    manifest = self.manifest(endpointer=dataset_endpointer)
    return manifest.dataset.table(name=endpointer.table_name)

  def layout(self, endpointer):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.LAYOUT:
      raise EndpointError('endpoint must be layout: %s' % (endpointer,))
    dataset_endpointer = endpointer.dataset_endpointer
    manifest = self.manifest(endpointer=dataset_endpointer)
    return manifest.dataset.table(name=endpointer.table_name).layout(name=endpointer.layout_name)

  def label_column(self, endpointer, label_column_name):
    layout = self.layout(endpointer=endpointer)
    return layout.label_column(name=label_column_name)

  def layout_array_values(self, endpointer, array_name):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.LAYOUT:
      raise EndpointError('endpoint must be layout: %s' % (endpointer,))
    url = api_fetch.layout_array_url(
      user_name=endpointer.user_name,
      dataset_name=endpointer.dataset_name,
      table_name=endpointer.table_name,
      layout_name=endpointer.layout_name,
      array_name=array_name,
    )
    with self.open(url=url) as f:
      return lib.LayoutArray.read_array_values(f=f)

  def chunk_df(self, endpointer, filename):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.LAYOUT:
      raise EndpointError('endpoint must be layout: %s' % (endpointer,))
    url = api_fetch.layout_chunk_url(
      user_name=endpointer.user_name,
      dataset_name=endpointer.dataset_name,
      table_name=endpointer.table_name,
      layout_name=endpointer.layout_name,
      filename=filename,
    )
    with self.open(url=url) as f:
      return pd.read_csv(f)

  def chunk_dfs(self, endpointer, filename, page_size=1_000, first_page_size=None):
    endpointer = lib.parse_endpointer(endpointer=endpointer)
    if endpointer.endpoint_type != lib.LAYOUT:
      raise EndpointError('endpoint must be layout: %s' % (endpointer,))
    url = api_fetch.layout_chunk_url(
      user_name=endpointer.user_name,
      dataset_name=endpointer.dataset_name,
      table_name=endpointer.table_name,
      layout_name=endpointer.layout_name,
      filename=filename,
    )
    if first_page_size is not None:
      # The first page should have a special size; all remaining pages have page_size rows.
      with self.open(url=url) as f:
        yield pd.read_csv(f, nrows=first_page_size)
      with self.open(url=url) as f:
        try:
          yield from pd.read_csv(f, skiprows=first_page_size, chunksize=page_size)
        except pd.errors.EmptyDataError:
          # This filename contained <= first_page_size rows; nothing more to read.
          pass
    else:
      # All pages have page_size rows.
      with self.open(url=url) as f:
        yield from pd.read_csv(f, chunksize=page_size)

  def query(self, sequence_filters):
    return queries.Query(sequence_filters=sequence_filters)
