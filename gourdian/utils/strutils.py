NUMBER_UNITS = ('', 'K', 'M', 'B', 'T')
SI_UNITS = ('', 'K', 'M', 'G', 'T')
BYTE_UNITS = ('B', 'KB', 'MB', 'GB', 'TB')


def format_number(num, units=NUMBER_UNITS):
  if abs(num) > 1_000_000_000_000:
    pretty, symbol = (num / 1_000_000_000_000), units[4]
  elif abs(num) > 1_000_000_000:
    pretty, symbol = (num / 1_000_000_000), units[3]
  elif abs(num) > 1_000_000:
    pretty, symbol = (num / 1_000_000), units[2]
  elif abs(num) > 1_000:
    pretty, symbol = (num / 1_000), units[1]
  else:
    return '%d%s' % (int(num), units[0])
  if pretty >= 100:
    return '%d%s' % (pretty, symbol)
  if pretty >= 10:
    return '%0.01f%s' % (pretty, symbol)
  return '%0.02f%s' % (pretty, symbol)
