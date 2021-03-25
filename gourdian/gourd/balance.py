import bisect
import collections
import itertools
import logging
import math
import multiprocessing
import os.path
import pathlib
import shutil
import tqdm
from concurrent import futures

from gourdian.utils import csvutils
from gourdian.utils import pdutils

NUM_CPUS = multiprocessing.cpu_count()


def human_format(num):
  # https://stackoverflow.com/a/45478574/195125
  units = ('', 'K', 'M', 'G', 'T', 'P')
  k = 1000.0
  magnitude = int(math.floor(math.log(num, k)))
  return '%0.02f%s' % (num / k**magnitude, units[magnitude])


FileSplit = collections.namedtuple('FileSplit', ('group_num', 'requires_split', 'size_range'))


def weighted_tqdm(items, weights=None, default_weight=0, total=None, **kwargs):
  weights = pdutils.coalesce(weights, {})
  total = total if not pdutils.is_empty(total) else (sum(weights.values()) if weights else None)
  if not weights and not default_weight:
    yield from items
  else:
    with tqdm.tqdm(items, total=total, **kwargs) as tq:
      for item in items:
        yield item
        weight = weights.get(item, default_weight)
        tq.update(weight)


class DummyDict:
  def __init__(self, keys, value=1):
    self.keys = keys
    self.value = value

  def __getitem__(self, key):
    return self.value

  def values(self):
    return itertools.repeat(self.value, len(self.keys))


class Allocation:
  __slots__ = ['groups', 'group_size', 'num_files']

  def __init__(self, num_groups, num_files):
    self.groups = [Group() for i in range(num_groups)]
    self.group_size = [0] * num_groups
    self.num_files = num_files

  def __repr__(self):
    return '<Allocation splits=%d (files=%d/chunks=%d) %r>' % (
      self.num_splits, self.num_files, self.num_chunks, dict(enumerate(self.groups)),)

  @property
  def num_chunks(self):
    return sum(len(group.chunks) for group in self.groups)

  @property
  def num_splits(self):
    return self.num_chunks - self.num_files

  def file_splits(self):
    group_chunks = sorted(((c, i) for i, s in enumerate(self.groups) for c in s.chunks),
                          key=lambda x: (str(x[0].path), x[0].size_range[0]))
    file_splits = collections.defaultdict(list)
    for chunk, group_num in group_chunks:
      file_splits[chunk.path].append(FileSplit(
        group_num=group_num,
        requires_split=chunk.requires_split,
        size_range=chunk.size_range,
      ))
    return file_splits

  def to_js(self):
    return {
      'num_files': self.num_files,
      'num_splits': self.num_splits,
      'num_chunks': self.num_chunks,
      'files': {str(k): [s._asdict() for s in v] for k, v in self.file_splits().items()},
      'groups': [g.to_js() for g in self.groups],
    }


class Group:
  __slots__ = ['chunks', 'chunk_size']

  def __init__(self):
    self.chunks = []
    self.chunk_size = []

  def __repr__(self):
    return '<Group %s chunks=%d>' % (human_format(num=sum(self.chunk_size)), len(self.chunks))

  def to_js(self):
    return {
      'chunks': [c.to_js() for c in self.chunks],
    }


class Chunk:
  __slots__ = ['path', 'size_range', 'total_size', 'requires_split']

  def __init__(self, path, size_range, total_size=None, requires_split=False):
    self.path = path
    self.size_range = size_range
    self.total_size = total_size if total_size is not None else (size_range[1] - size_range[0])
    self.requires_split = requires_split

  def __repr__(self):
    return '<Chunk %s %r [%d:%d]>' % (human_format(num=self.total_size), self.path.name,
                                      self.size_range[0], self.size_range[1])

  def split(self):
    path = self.path
    range_head, range_bomb = self.size_range
    range_mid = range_head + int(self.total_size / 2)
    return [
      Chunk(
        path=path,
        size_range=(range_head, range_mid),
        total_size=(range_mid - range_head),
        requires_split=True,
      ),
      Chunk(
        path=path,
        size_range=(range_mid, range_bomb),
        total_size=(range_bomb - range_mid),
        requires_split=True,
      ),
    ]

  def to_js(self):
    return {
      'path': str(self.path),
      'size_range': self.size_range,
      'total_size': self.total_size,
      'requires_split': self.requires_split,
    }


def csv_path_rows(csv_paths, has_header=True, csv_weights=1):
  """Yields (path, num_rows) for an iterable of csv_paths.

  If provided, csv_weights is used to update the progress bar with precise costs.
  """
  # TODO(rob): create some kind of dummy tqdm bar.
  csv_paths = tuple(csv_paths)
  if isinstance(csv_weights, int):
    csv_weights = DummyDict(keys=csv_paths, value=csv_weights)
  if csv_weights is not None:
    total = sum(csv_weights.values())
    tq = tqdm.tqdm(desc='counting csv rows', total=total, leave=False)
  else:
    tq = None
  for csv_path in csv_paths:
    tq and tq.update(csv_weights[csv_path])
    yield csv_path, csvutils.num_rows(csv_path=csv_path, has_header=has_header)
  tq and tq.close()


def csv_path_bytes(csv_paths, csv_weights=1):
  csv_paths = tuple(csv_paths)
  if isinstance(csv_weights, int):
    csv_weights = DummyDict(keys=csv_paths, value=csv_weights)
  if csv_weights is not None:
    total = sum(csv_weights.values())
    tq = tqdm.tqdm(desc='counting csv bytes', total=total, leave=False)
  else:
    tq = None
  for csv_path in csv_paths:
    tq and tq.update(csv_weights[csv_path])
    yield csv_path, csv_path.stat().st_size
  tq and tq.close()


def merge(csv_paths, f_out, header=None, has_headers=True, csv_weights=1):
  csv_paths = tuple(csv_paths)
  weights = csv_weights if hasattr(csv_weights, '__getitem__') else None
  default_weight = 0 if hasattr(csv_weights, '__getitem__') else csv_weights
  csv_paths = weighted_tqdm(csv_paths, weights=weights, default_weight=default_weight,
                            desc='merging csvs')
  return csvutils.merge(csv_paths=csv_paths, f_out=f_out, header=header, has_headers=has_headers,
                        errors=csvutils.LOG)


def allocate(csv_path_size, num_groups):
  def insert_chunk(allocation, chunk):
    # The smallest group is in groups[0].
    group = allocation.groups.pop(0)
    group_size = allocation.group_size.pop(0)
    # Insert chunk into the smallest group in the correct sort order (smallest-largest chunks).
    chunk_index = bisect.bisect_left(group.chunk_size, chunk.total_size)
    group.chunks.insert(chunk_index, chunk)
    group.chunk_size.insert(chunk_index, chunk.total_size)
    # Reinsert group back into the allocation in the correct sort order (smallest-largest groups).
    group_size = group_size + chunk.total_size
    group_index = bisect.bisect_left(allocation.group_size, group_size)
    allocation.groups.insert(group_index, group)
    allocation.group_size.insert(group_index, group_size)

  def optimize(allocation, threshold_percent=0.1):
    while True:
      diff_size = (allocation.group_size[-1] - allocation.group_size[0])
      diff_percent = diff_size / (allocation.group_size[0] or diff_size)
      if diff_percent <= threshold_percent:
        break
      # The largest group is in groups[-1].
      largest_group = allocation.groups.pop(-1)
      largest_group_size = allocation.group_size.pop(-1)
      # Remove the largest chunk from the largest group.
      largest_chunk = largest_group.chunks.pop(-1)
      _ = largest_group.chunk_size.pop(-1)
      # Re-insert the (formerly) largest_group back where it belongs.
      # TODO(rob): insert_chunk pointlessly pops and re-inserts this again immediately.
      largest_group_size = largest_group_size - largest_chunk.total_size
      group_index = bisect.bisect_left(allocation.group_size, largest_group_size)
      allocation.groups.insert(group_index, largest_group)
      allocation.group_size.insert(group_index, largest_group_size)
      # Cut the largest chunk it into 2, and re-insert it.
      head_chunk, bomb_chunk = largest_chunk.split()
      insert_chunk(allocation=allocation, chunk=head_chunk)
      insert_chunk(allocation=allocation, chunk=bomb_chunk)

  # Create initial allocation of whole csv_path to groups.
  num_groups = pdutils.coalesce(num_groups, NUM_CPUS)
  allocation = Allocation(num_groups=num_groups, num_files=0)
  for csv_path, csv_size in csv_path_size:
    allocation.num_files += 1
    csv_range = (0, csv_size)
    chunk = Chunk(path=csv_path, size_range=csv_range, total_size=csv_size)
    insert_chunk(allocation=allocation, chunk=chunk)
  # Optimize groups by halving the largest chunk from the largest group.
  optimize(allocation=allocation)
  return allocation


def _group_out_path(group_num, path, out_dir, relative_to, fmt='%d'):
  path = pathlib.Path(path)
  out_dir = pathlib.Path(out_dir)
  group_name = fmt % (group_num,)
  return out_dir/group_name/path.relative_to(relative_to)


def _write_groups(out_dir, relative_to, path, file_splits, num_groups, symlink_ok=True):
  fmt = '%%0%dd' % (len(str(num_groups)),)
  if (len(file_splits) == 1) and file_splits[0].requires_split is False:
    group_num, _, _ = file_splits[0]
    out_path = _group_out_path(group_num=group_num, path=path, out_dir=out_dir,
                               relative_to=relative_to, fmt=fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_ok:
      os.symlink(src=path, dst=out_path)
    else:
      shutil.copy2(path, out_path)
  else:
    # Convert ranges to bytes_per_split/rows_per_split; the final split should consume all remaning.
    sizes_per_split = [sr[1] - sr[0] for (_, _, sr) in file_splits]
    sizes_per_split[-1] = None
    out_dir = [_group_out_path(group_num=sn, path=path, out_dir=out_dir, relative_to=relative_to,
                               fmt=fmt)
               for (sn, _, _) in file_splits]
    assert len(sizes_per_split) == len(out_dir)
    with open(path, 'r') as csv_in:
      csvutils.split_by_bytes(csv_in=csv_in, bytes_per_split=sizes_per_split, out_dir=out_dir)
  return str(path)


def balance(csv_paths, out_dir, num_groups=None, max_workers=None, relative_to=None,
            symlink_ok=True, csv_path_weights=None):
  """Computes an allocation and materializes it by splitting large files and symlinking others.

  This function creates roughly equal-sized directories of csvs inside `out_dir`, in preparation for
  processing them in parallel.

  Confusingly, the `num_groups` parameter specifies the number of directories to allocate csvs into,
  while the `max_workers` parameter specifies the number of workers to use while materializing the
  allocation.

  csv_paths: iterable of csv file paths
  out_dir: non-existant directory to write new files into
  num_groups: number of groups to allocate files into; default max_workers
  max_workers: max number of parallel processes to run; default NUM_CPUS
  relative_to: drop this prefix when writing files into out_dir; default csv_paths common prefix
  symlink_ok: if True, symlinked non-split files into out_dir; if False, copy files
  csv_path_weights {path: int}: optional mapping of path to allocation weight; default csv bytes
  """
  max_workers = pdutils.coalesce(max_workers, NUM_CPUS)
  num_groups = pdutils.coalesce(num_groups, max_workers)
  # Read the number of rows in each csv file, and allocate them into num_groups.
  csv_paths = tuple(csv_paths)
  if csv_path_weights is None:
    csv_path_weights = dict(csv_path_bytes(csv_paths=csv_paths))
  logging.info('allocating csvs into %s groups...' % num_groups)
  allocation = allocate(csv_path_size=tuple(csv_path_weights.items()), num_groups=num_groups)
  logging.info(allocation)
  # Perform all required splits into their split_num directory.
  relative_to = relative_to or os.path.commonpath(csv_paths)
  file_splits = allocation.file_splits()
  job_kw = {'out_dir': str(out_dir), 'relative_to': str(relative_to), 'num_groups': num_groups,
            'symlink_ok': symlink_ok}
  jobs = ((_write_groups, dict(path=str(path), file_splits=fs, **job_kw))
          for path, fs in file_splits.items())
  with tqdm.tqdm(desc='splitting', total=sum(csv_path_weights.values()),
                 mininterval=1, maxinterval=1) as tq:
    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
      pending = set(pool.submit(fn, **kw) for fn, kw in itertools.islice(jobs, max_workers * 2))
      while pending:
        done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
        for job in done:
          finished_path = pathlib.Path(job.result())
          tq.update(csv_path_weights[finished_path])
        for fn, kw in itertools.islice(jobs, len(done)):
          pending.add(pool.submit(fn, **kw))
  return allocation
