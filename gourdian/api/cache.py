import hashlib
import io
import pathlib
import urllib.parse


DEFAULT_CACHE_DIR = '~/.gourdian'


class DummyCache:
  def get(self, url):
    return None

  def put(self, url, response):
    return io.BytesIO(response.content)


class LocalCache:
  def __init__(self, cache_dir=DEFAULT_CACHE_DIR):
    self._cache_dir = pathlib.Path(cache_dir).expanduser().absolute()

  def __repr__(self):
    return '<%s %r>' % (self.__class__.__name__, str(self.cache_dir))

  def __str__(self):
    return str(self.cache_dir)

  @property
  def cache_dir(self):
    return self._cache_dir

  def cache_path(self, url):
    scheme, netloc, path, query, _ = urllib.parse.urlsplit(url)
    if not path:
      raise ValueError('cannot cache bare domain url: %s' % (url,))
    domain_str = '%s|%s' % (scheme, netloc)
    domain_str = domain_str.strip('/')
    path_str = '%s|%s' % (path, hashlib.md5(query).hexdigest()) if query else path
    path_str = path_str.strip('/')
    return self.cache_dir/domain_str/path_str

  def get(self, url):
    local_path = self.cache_path(url=url)
    if local_path.is_file():
      return local_path.open('rb')

  def put(self, url, response):
    local_path = self.cache_path(url=url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open('wb') as f:
      for chunk in response.iter_content():
        f.write(chunk)
    return self.get(url=url)
