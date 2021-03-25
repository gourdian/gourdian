import csv
import itertools
import pathlib
import sys


RAISE = 'raise'
LOG = 'log'


def read_header(csv_path, errors=RAISE):
  with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    return next(reader)


def extract_header(csv_paths, errors=RAISE):
  headers = set(x[1] for x in extract_headers(csv_paths=csv_paths, errors=errors))
  if len(headers) == 0:
    raise ValueError('found no csv headers')
  if len(headers) > 1:
    raise ValueError('found multiple csv headers: %d' % (len(headers),))
  return next(iter(headers))


def extract_headers(csv_paths, errors=RAISE):
  for csv_path in csv_paths:
    with open(csv_path, 'r') as f:
      reader = csv.reader(f)
      try:
        header = next(reader)
        yield csv_path, tuple(header)
      except StopIteration:
        # Empty csv.
        yield csv_path, tuple()
      except Exception:
        if errors == RAISE:
          raise IOError('could not read %s' % (csv_path,))
        sys.stderr.write('could not read: %s\n' % (csv_path,))


def num_rows(csv_path, has_header=True, errors=RAISE):
  with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    if has_header:
      _ = next(reader)
    return sum(1 for x in reader)


def split_by_rows(csv_in, out_dir, rows_per_split):
  """Split csv_in into multiple files of rows_per_split rows; write to out_dir.

  rows_per_split:
    - if int: create every split with this many rows
    - if (int, ...): create splits with these int rows; None to consume all remaining
  """
  # Convert scalar out_dir/bytes_per_split to repeating iterables.
  if isinstance(out_dir, (str, pathlib.Path)):
    out_dir = itertools.repeat(out_dir)
  if isinstance(rows_per_split, int):
    rows_per_split = itertools.repeat(rows_per_split)
  # Split csv_in into chunks of approximately bytes_per_split.
  reader = csv.reader(csv_in)
  header = next(reader)
  rows_consumed = 0
  for num_rows, out in zip(rows_per_split, out_dir):
    rows_break = None if num_rows is None else rows_consumed + num_rows
    out_fname = 'rows.%d.%s.csv' % (rows_consumed, str(rows_break) if rows_break else 'eof')
    out_path = pathlib.Path(out)/out_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    # logging.info('writing %s...', out_path)
    with out_path.open('w') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for row in reader:
        writer.writerow(row)
        rows_written += 1
        if (num_rows is not None) and (rows_written >= num_rows):
          # If num_rows is None: consume all remaining rows (never break).
          rows_consumed += rows_written
          break
    if (num_rows is None) or (rows_written < num_rows):
      # All remaining rows have been consumed.
      return


def split_by_bytes(csv_in, out_dir, bytes_per_split):
  """Split csv_in into multiple files of approx bytes_per_split size; write to out_dir.

  bytes_per_split:
    - if int: create every split with this filesize
    - if (int, ...): create splits with these int filesizes; None to consume all remaining
  """
  # Convert scalar out_dir/bytes_per_split to repeating iterables.
  if isinstance(out_dir, (str, pathlib.Path)):
    out_dir = itertools.repeat(out_dir)
  if isinstance(bytes_per_split, int):
    bytes_per_split = itertools.repeat(bytes_per_split)
  # Split csv_in into chunks of approximately bytes_per_split.
  reader = csv.reader(csv_in)
  header = next(reader)
  bytes_consumed = 0
  rows_consumed = 1
  for num_bytes, out in zip(bytes_per_split, out_dir):
    bytes_break = None if num_bytes is None else bytes_consumed + num_bytes
    out_fname = 'rows.%d.csv' % (rows_consumed,)
    out_path = pathlib.Path(out)/out_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    # logging.info('writing %s...', out_path)
    with out_path.open('w') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for row in reader:
        writer.writerow(row)
        rows_consumed += 1
        if (num_bytes is not None) and (f.tell() >= num_bytes):
          # If num_bytes is None: consume all remaining bytes (never break).
          bytes_consumed = bytes_break
          bytes_written = f.tell()
          break
    if (num_bytes is None) or (bytes_written == 0):
      # All remaining rows have been consumed.
      return


def merge(csv_paths, f_out, header=None, has_headers=True, errors=RAISE):
  csv_paths = tuple(csv_paths)
  writer = csv.writer(f_out)
  if header is not None:
    # Write the header that was explicitly provided.
    writer.writerow(header)
  for csv_path in csv_paths:
    with open(csv_path, 'r') as f_in:
      reader = csv.reader(f_in)
      if has_headers:
        path_header = tuple(next(reader))
        if header is None:
          # An explicit header was not provided; pull it from the first file in csv_paths.
          header = path_header
          writer.writerow(header)
        else:
          assert header == path_header, 'found header mismatch: %s' % (csv_path,)
      for row in reader:
        writer.writerow(row)
