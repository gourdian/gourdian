import functools


def to_tuple(fn):
  @functools.wraps(fn)
  def helper(*args, **kwargs):
    return tuple(fn(*args, **kwargs))

  return helper
