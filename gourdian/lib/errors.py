COERCE = 'coerce'
DROP = 'drop'
IGNORE = 'ignore'
RAISE = 'raise'


def validate_errors(errors, coerce_ok=False, drop_ok=False, ignore_ok=False, raise_ok=False):
  valid_errors = [coerce_ok and COERCE, drop_ok and DROP, ignore_ok and IGNORE, raise_ok and RAISE]
  valid_errors = tuple(x for x in valid_errors if x)
  if errors not in valid_errors:
    raise ValueError('errors must be in %r: %r' % (valid_errors, errors))


class AlreadyBoundError(Exception):
  pass


class DuplicateNameError(Exception):
  pass


class NoLabelColumnsError(Exception):
  pass


class MultipleLabelColumnsError(Exception):
  pass
