#! /usr/bin/env python3

import argparse
import ast
import collections
import colored
import csv
import itertools
import json
import logging
import math
import pathlib
import sys

from gourdian.gourd import balance
from gourdian.gourd import ui
from gourdian.utils import csvutils


def colorize(text, fg):
  if sys.stdout.isatty():
    return colored.stylize(text, colored.fg(fg))
  return text


def extract_csv_paths(csv_paths, glob='**/*.csv'):
  for csv_path in sorted(csv_paths):
    csv_path = pathlib.Path(csv_path)
    if csv_path.is_dir():
      yield from sorted(x for x in csv_path.glob(glob) if x.is_file())
    else:
      yield csv_path


def main_show(csv_in, has_header):
  ui.main(csv_in=csv_in, has_header=has_header)


def main_headers(csv_paths, filename_ok=True, count_ok=False):
  csv_paths = extract_csv_paths(csv_paths)
  headers = csvutils.extract_headers(csv_paths=csv_paths, errors='log')
  writer = csv.writer(sys.stdout)
  if count_ok:
    headers = sorted(collections.Counter(x[1] for x in headers).items(), key=lambda x: -x[1])
    for header, count in headers:
      sys.stdout.write(colorize('%s: ' % (count,), fg='cyan'))
      writer.writerow(header)
  else:
    for csv_path, header in headers:
      if filename_ok:
        sys.stdout.write(colorize('%s: ' % (csv_path,), fg='cyan'))
      writer.writerow(header)


def main_header(csv_paths):
  if csv_paths and len(csv_paths) == 1 and hasattr(csv_paths[0], 'read'):
    header = csvutils.read_header(csv_in=csv_paths[0])
  else:
    csv_paths = extract_csv_paths(csv_paths)
    header = csvutils.extract_header(csv_paths)
  writer = csv.writer(sys.stdout)
  writer.writerow(header)


def main_rename(csv_in, csv_out, header, lacks_header):
  def update_header(old_header, replace):
    for index, new_name in replace.items():
      if not isinstance(index, int):
        index = old_header.index(index)
      old_header[index] = new_name

  reader = csv.reader(csv_in)
  writer = csv.writer(csv_out)
  try:
    first_row = next(reader)
  except StopIteration:
    # File is empty.
    raise
  # Read the first row of file and create a header template (old_header) from it.
  if lacks_header:
    old_header = ['' for _ in range(len(first_row))]
  else:
    old_header = first_row
  # Update old_header with info from header.
  try:
    replace = ast.literal_eval(header)
    if isinstance(replace, list):
      writer.writerow(replace)
    else:
      update_header(old_header=old_header, replace=replace)
      writer.writerow(old_header)
  except RuntimeError:
    # Provided header was a non-json string; use it unchanged.
    csv_out.write(header)
    csv_out.write('\n')
  # Write the remainder of file into out.
  if lacks_header:
    # The first row was used to create an empty header, but it is also a data row!
    writer.writerow(first_row)
  for row in reader:
    writer.writerow(row)


def main_cut(csv_in, csv_out, cut_cols):
  reader = csv.reader(csv_in)
  writer = csv.writer(csv_out)
  header = next(reader)
  cut_cols = ast.literal_eval(cut_cols)
  cut_indexes = set(x if isinstance(x, int) else header.index(x) for x in cut_cols)
  # Write header.
  writer.writerow([x for i, x in enumerate(header) if i not in cut_indexes])
  # Write all rows.
  for row in reader:
    writer.writerow([x for i, x in enumerate(row) if i not in cut_indexes])


def main_head(csv_in, out, n):
  reader = csv.reader(csv_in)
  header = next(reader)
  rows = itertools.islice(reader, n)
  writer = csv.writer(out)
  writer.writerow(header)
  for row in rows:
    writer.writerow(row)


def main_count(csv_paths, has_header):
  def count_csv(csv_in, has_header):
    reader = csv.reader(csv_in)
    try:
      if has_header:
        _ = next(reader)
    except StopIteration:
      return 0
    return sum(1 for x in reader)

  if not csv_paths or csv_paths == ['-']:
    num_rows = count_csv(csv_in=sys.stdin, has_header=has_header)
    print('%d' % (num_rows,))
  else:
    csv_paths = extract_csv_paths(csv_paths)
    total_num_rows = 0
    num_csvs = 0
    for csv_path in csv_paths:
      with csv_path.open('r') as csv_in:
        num_rows = count_csv(csv_in=csv_in, has_header=has_header)
        sys.stderr.write('%8d' % (num_rows,))
        sys.stderr.write(colorize(' %s\n' % (csv_path,), fg='cyan'))
        total_num_rows += num_rows
      num_csvs += 1
    if num_csvs > 1:
      print('%8d total' % (total_num_rows,))


def main_split(csv_in, out_dir, num_splits=None, bytes_per_split=None, rows_per_split=None):
  # Ensure we have a clean output directory before we do any expensive work.
  out_dir = pathlib.Path(out_dir)
  pathlib.Path(out_dir).mkdir(parents=True, exist_ok=False)
  # Features depend on whether csv_in is a regular file or not.
  if csv_in.seekable() and num_splits is not None:
    csv_path = pathlib.Path(csv_in.name).absolute()
    csv_bytes = csv_path.stat().st_size
    # NOTE: [None] as the final value means all remaining rows will be consumed by the final split.
    bytes_per_split = ([math.ceil(csv_bytes / num_splits)] * (num_splits-1)) + [None]
  if bytes_per_split is not None:
    return csvutils.split_by_bytes(csv_in=csv_in, out_dir=out_dir, bytes_per_split=bytes_per_split)
  if rows_per_split is not None:
    return csvutils.split_by_rows(csv_in=csv_in, out_dir=out_dir, rows_per_split=rows_per_split)
  raise ValueError('must give bytes_per_split or rows_per_split: %r, %r' %
                   (bytes_per_split, rows_per_split))


def main_merge(csv_paths, out):
  csv_paths = extract_csv_paths(csv_paths)
  if isinstance(out, str):
    out = pathlib.Path(out)
  if hasattr(out, 'write'):
    balance.merge(csv_paths=csv_paths, f_out=out)
  else:
    with open(out, 'w') as f_out:
      balance.merge(csv_paths=csv_paths, f_out=f_out)


def main_allocate(csv_paths, num_groups=None, summary_ok=True):
  csv_paths = extract_csv_paths(csv_paths=csv_paths)
  csv_path_size = balance.csv_path_bytes(csv_paths=csv_paths, csv_weights=1)
  allocation = balance.allocate(csv_path_size=csv_path_size, num_groups=num_groups)
  if summary_ok:
    print(allocation)
  else:
    json.dump(allocation.to_js(), sys.stdout)


def main_balance(csv_paths, out_dir, symlink_ok, num_groups=None):
  # Ensure we have a clean output directory before we do any expensive work.
  out_dir = pathlib.Path(out_dir)
  pathlib.Path(out_dir).mkdir(parents=True)
  # Split files will live in `out_dir/$RELATIVE_TO/bytes.%d.%d.csv`.
  if len(csv_paths) == 1 and pathlib.Path(csv_paths[0]).is_dir():
    relative_to = pathlib.Path(csv_paths[0])
  else:
    relative_to = None
  csv_paths = extract_csv_paths(csv_paths=csv_paths)
  balance.balance(csv_paths=csv_paths, out_dir=out_dir, num_groups=num_groups,
                  relative_to=relative_to, symlink_ok=symlink_ok)


def main():
  logging.basicConfig(format='%(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='action', required=True)
  # show
  parser_show = subparsers.add_parser('show')
  parser_show.add_argument('csv_in', type=argparse.FileType('r'), default='-', nargs='?')
  show_header_group = parser_show.add_mutually_exclusive_group(required=False)
  show_header_group.add_argument('--no-header', action='store_false', dest='has_header')
  show_header_group.add_argument('--has-header', action='store_true', dest='has_header')
  parser_show.set_defaults(func=main_show, has_header=True)
  # headers
  parser_headers = subparsers.add_parser('headers')
  parser_headers.add_argument('csv_paths', type=str, nargs='+')
  parser_headers.add_argument('--no-filename', action='store_false', default=True,
                              dest='filename_ok')
  parser_headers.add_argument('--count', action='store_true', default=False, dest='count_ok')
  parser_headers.set_defaults(func=main_headers)
  # header
  parser_header = subparsers.add_parser('header')
  parser_header.add_argument('csv_paths', nargs='*', type=argparse.FileType('r'),
                             default=[sys.stdin])
  parser_header.set_defaults(func=main_header)
  # rename
  parser_rename = subparsers.add_parser('rename')
  parser_rename.add_argument('header')
  parser_rename.add_argument('csv_in', type=argparse.FileType('r'), default=sys.stdin, nargs='?')
  parser_rename.add_argument('csv_out', type=argparse.FileType('w'), default=sys.stdout, nargs='?')
  parser_rename.add_argument('--lacks-header', default=False, action='store_true')
  parser_rename.set_defaults(func=main_rename)
  # cut
  parser_cut = subparsers.add_parser('cut')
  parser_cut.add_argument('cut_cols')
  parser_cut.add_argument('csv_in', type=argparse.FileType('r'), default=sys.stdin, nargs='?')
  parser_cut.add_argument('csv_out', type=argparse.FileType('w'), default=sys.stdout, nargs='?')
  parser_cut.set_defaults(func=main_cut)
  # head
  parser_head = subparsers.add_parser('head')
  parser_head.add_argument('csv_in', type=argparse.FileType('r'), default='-', nargs='?')
  parser_head.add_argument('out', type=argparse.FileType('w'), default='-', nargs='?')
  parser_head.add_argument('-n', type=int, default=10)
  parser_head.set_defaults(func=main_head)
  # count
  parser_count = subparsers.add_parser('count')
  parser_count.add_argument('csv_paths', type=str, nargs='*')
  count_header_group = parser_count.add_mutually_exclusive_group(required=False)
  count_header_group.add_argument('--no-header', action='store_false', dest='has_header')
  count_header_group.add_argument('--has-header', action='store_true', dest='has_header')
  parser_count.set_defaults(func=main_count, has_header=True)
  # split
  parser_split = subparsers.add_parser('split')
  parser_split.add_argument('csv_in', type=argparse.FileType('r'), default='-', nargs='?')
  parser_split.add_argument('out_dir', type=str, default='.')
  split_size_group = parser_split.add_mutually_exclusive_group(required=True)
  split_size_group.add_argument('--splits', type=int, dest='num_splits')
  split_size_group.add_argument('--bytes', type=int, dest='bytes_per_split')
  split_size_group.add_argument('--rows', type=int, dest='rows_per_split')
  parser_split.set_defaults(func=main_split)
  # merge
  parser_merge = subparsers.add_parser('merge')
  parser_merge.add_argument('csv_paths', type=str, nargs='+')
  parser_merge.add_argument('out', type=argparse.FileType('w'), default='-', nargs='?')
  parser_merge.set_defaults(func=main_merge)
  # allocate
  parser_allocate = subparsers.add_parser('allocate')
  parser_allocate.add_argument('csv_paths', type=str, nargs='+')
  parser_allocate.add_argument('--groups', type=int, dest='num_groups')
  allocate_output_group = parser_allocate.add_mutually_exclusive_group(required=False)
  allocate_output_group.add_argument('--summary', action='store_true', dest='summary_ok')
  allocate_output_group.add_argument('--json', action='store_false', dest='summary_ok')
  parser_allocate.set_defaults(func=main_allocate, summary_ok=True)
  # balance
  parser_balance = subparsers.add_parser('balance')
  parser_balance.add_argument('csv_paths', type=str, nargs='+')
  parser_balance.add_argument('out_dir', type=str, default='.')
  parser_balance.add_argument('--groups', type=int, dest='num_groups')
  balance_copy_group = parser_balance.add_mutually_exclusive_group(required=False)
  balance_copy_group.add_argument('--copy', action='store_false', dest='symlink_ok')
  balance_copy_group.add_argument('--symlink', action='store_true', dest='symlink_ok')
  parser_balance.set_defaults(func=main_balance, symlink_ok=True)
  # Run the requested action.
  args = parser.parse_args()
  kwargs = dict((k, v) for k, v in args._get_kwargs() if k not in ('func', 'action'))
  return args.func(**kwargs)


if __name__ == '__main__':
  main()
