def format_number(num):
  if abs(num) > 1_000_000_000_000:
    pretty, symbol = (num / 1_000_000_000_000), 'T'
  elif abs(num) > 1_000_000_000:
    pretty, symbol = (num / 1_000_000_000), 'B'
  elif abs(num) > 1_000_000:
    pretty, symbol = (num / 1_000_000), 'M'
  elif abs(num) > 1_000:
    pretty, symbol = (num / 1_000), 'K'
  else:
    return '%d' % (int(num),)
  if pretty >= 100:
    return '%d%s' % (pretty, symbol)
  if pretty >= 10:
    return '%0.01f%s' % (pretty, symbol)
  return '%0.02f%s' % (pretty, symbol)
