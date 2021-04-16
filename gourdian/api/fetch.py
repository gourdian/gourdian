import logging
import pandas as pd
import requests


DEFAULT_URL_ROOT = 'https://data.gourdian.net/files/datasets'


def fetch(url):
  logging.info('GET %r', url)
  response = requests.get(url)
  response.raise_for_status()
  return response


def to_url(path, url_root=DEFAULT_URL_ROOT):
  return '%s/%s' % (url_root.rstrip('/'), path.lstrip('/'))


def manifest_filename(gz_ok=True):
  return 'manifest.json.gz' if gz_ok else 'manifest.json'


def manifest_url(user_name, dataset_name, url_root=DEFAULT_URL_ROOT):
  path = '/%s/%s/%s' % (user_name, dataset_name, manifest_filename())
  return to_url(path=path, url_root=url_root)


def layout_array_filename(name, gz_ok=True):
  """Returns the filename for an array with a given name."""
  suffix = '.bin.gz' if gz_ok else '.bin'
  return 'array|%s%s' % (name, suffix)


def layout_array_url(user_name, dataset_name, table_name, layout_name, array_name,
                     url_root=DEFAULT_URL_ROOT):
  array_filename = layout_array_filename(name=array_name)
  path = '/%s/%s/%s/%s/arrays/%s' % (user_name, dataset_name, table_name, layout_name,
                                     array_filename)
  return to_url(path=path, url_root=url_root)


def layout_chunk_filename(label=(), gz_ok=True):
  """Returns the chunk filename for an iterable of properly encoded label strings."""
  if ((isinstance(label, pd.Series) and label.isna().any())
      or (None in label)):
    return None
  label_str = '|'.join(label)
  return 'chunk|%s%s' % (label_str, '.csv.gz' if gz_ok else '.csv')


def layout_chunk_url(user_name, dataset_name, table_name, layout_name, filename,
                     url_root=DEFAULT_URL_ROOT):
  path = '/%s/%s/%s/%s/chunks/%s' % (user_name, dataset_name, table_name, layout_name, filename)
  return to_url(path=path, url_root=url_root)
