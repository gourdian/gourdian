import asyncio
import json
import os
import pandas as pd
import sys
import termios
import urwid
from urwid import signals

from gourdian.utils import pdutils


PALETTE = [
  ('column-header', 'dark gray,bold', 'default'),
  ('column-header-selected', 'default,bold,standout', 'default'),
  ('column-header-hilighted', 'default,bold', 'default'),
  ('row-item', 'default', 'default'),
  ('row-item-selected', 'default,standout', 'default'),
  ('row-item-hilighted', 'default', 'dark gray'),
  ('row-item-nil', 'dark cyan', 'default'),
  ('row-item-nil-selected', 'default,standout', 'dark cyan'),
  ('row-item-nil-hilighted', 'dark cyan', 'dark gray'),
  ('footer-header', 'standout', 'default'),
  ('footer-dim', 'dark gray', 'default'),
]


def fixterm():
  fd = sys.stdin.fileno()
  iflag, oflag, cflag, lflag, ispeed, ospeed, cc = termios.tcgetattr(fd)
  termios.tcsetattr(fd, termios.TCSANOW,
                    [iflag, oflag, cflag, lflag | termios.ECHO, ispeed, ospeed, cc])
  return [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]


def pd_align(obj):
  return 'right' if pd.api.types.is_numeric_dtype(obj) else 'left'


def unhandled_input(key):
  if key in ('q', 'Q'):
    raise urwid.ExitMainLoop()
  return False


class SelectableText(urwid.Text):
  _selectable = True

  def keypress(self, size, key):
    if key == 'enter':
      pass
    else:
      return key


class FlexibleColumns(urwid.Columns):
  def _erase_one(self, items):
    for i, item in enumerate(items):
      if item != 0:
        items[i] = 0
        return

  def column_widths(self, size, focus=False):
    maxcol = size[0]
    if maxcol == self._cache_maxcol:
      return self._cache_column_widths
    widths = []
    for widget, (width_type, width, _) in self.contents:
      assert width_type == 'given'
      widths.append(width + self.dividechars)
    while True:
      width_left = widths[0:self.focus_position+1]
      if sum(width_left) <= maxcol:
        break
      self._erase_one(widths)
    for i in range(len(widths)):
      if sum(widths[0:i+1]) > maxcol:
        widths = widths[0:i+1]
        widths[i] += (maxcol - sum(widths) + self.dividechars)
        break
    return [x - self.dividechars for x in widths]


class DataFrameWalker(urwid.ListWalker):
  def __init__(self, df, col_widths):
    self._df = df
    self._col_widths = col_widths
    self.focus = 0
    # Runtime state.
    self._focused_col = 0
    self._widgets = {}
    self._size = None

  def _cell(self, position, col_index, val, width):
    align = pd_align(self._df.iloc[:, col_index].dtype)
    is_hilighted = (self.focus == position)
    if pdutils.is_empty(val):
      widget = urwid.AttrMap(SelectableText(str(val), wrap='clip', align=align),
                             'row-item-nil-hilighted' if is_hilighted else 'row-item-nil',
                             'row-item-nil-selected')
    else:
      widget = urwid.AttrMap(SelectableText(str(val).replace('\n', '\\n').replace('\r', '\\r'),
                                            wrap='clip', align=align),
                             None, 'row-item-selected')
    return (width, widget)

  def __getitem__(self, position):
    df = self._df
    if 0 <= position < len(df):
      widget = self._widgets.get(position, None)
      if widget is None:
        cells = [self._cell(position=position, col_index=i, val=v, width=w)
                 for i, (v, w) in enumerate(zip(tuple(df.iloc[position]), self._col_widths))]
        columns = FlexibleColumns(cells, dividechars=1, focus_column=self._focused_col)
        widget = self._widgets[position] = urwid.AttrMap(columns, 'row-item', 'row-item-hilighted')
      return widget

  def set_focused_col(self, focused_col):
    self._focused_col = focused_col
    for row_widget in self._widgets.values():
      row_widget.base_widget.set_focus_column(self._focused_col)

  def get_focus(self):
    focus = self.focus
    return self[focus], focus

  def set_focus(self, focus):
    if self.focus in self._widgets:
      del self._widgets[self.focus]
    if focus in self._widgets:
      del self._widgets[focus]
    self.focus = focus
    self._modified()

  def next_position(self, position):
    if (len(self._df) - 1) <= position:
      raise IndexError
    return position + 1

  def prev_position(self, position):
    if position <= 0:
      raise IndexError
    return position - 1


class DataFrameListBox(urwid.ListBox):
  signals = ['focused_row_changed', 'focused_col_changed']

  def __init__(self, df, col_widths):
    self._size = None
    self._focused_col = None
    self._walker = DataFrameWalker(df=df, col_widths=col_widths)
    signals.connect_signal(
      self._walker,
      'modified',
      lambda: signals.emit_signal(self, 'focused_row_changed', self.focus_position),
    )
    super().__init__(self._walker)

  @property
  def focused_cell(self):
    focused_row = self.focus_position
    focused_col = self.focus.base_widget.focus_position
    return (focused_row, focused_col)

  def set_focused_col(self, focused_col):
    if focused_col != self._focused_col:
      self._focused_col = focused_col
      self.body.set_focused_col(focused_col=focused_col)
      signals.emit_signal(self, 'focused_col_changed', focused_col)

  def render(self, size, focus=False):
    self._size = size
    self.set_focused_col(focused_col=self.focus.base_widget.focus_position)
    return super().render(size=size, focus=focus)


class DataFrameTable(urwid.WidgetWrap):
  MAX_COL_WIDTH = 42

  @classmethod
  def _df_widths(cls, df):
    df = df.astype(str)
    row_max_len = tuple(df.apply(lambda sr: sr.str.strip().str.len().max()))
    return tuple(min(cls.MAX_COL_WIDTH, max(row_max_len[i], len(str(df.columns[i]))))
                 for i in range(len(df.columns)))

  def __init__(self, df, csv_path):
    self._df = df
    self._csv_path = csv_path if (csv_path and csv_path != '-') else '<stdin>'
    self._col_widths = self._df_widths(df=df.head(200))
    self._header = FlexibleColumns(
      [(w, urwid.AttrMap(urwid.Text(str(x), wrap='clip', align='left'),
                         'column-header', 'column-header-selected'))
       for i, (w, x) in enumerate(zip(self._col_widths, self._df.columns))],
      dividechars=1,
    )
    self._listbox = DataFrameListBox(df=df, col_widths=self._col_widths)
    self._footer_header = urwid.AttrMap(urwid.Columns([
      urwid.Text(str(self._csv_path), wrap='clip'),
      (8, urwid.Text('', wrap='clip', align='right')),
      (6, urwid.Text('', align='right')),
    ], dividechars=1), 'footer-header')
    self._footer_body = urwid.Text('', align='left', wrap='clip')
    super().__init__(urwid.Frame(
      header=urwid.Pile([self._header, urwid.Divider(div_char='─')]),
      body=self._listbox,
      footer=urwid.Pile([self._footer_header, self._footer_body]),
    ))
    signals.connect_signal(self._listbox, 'focused_col_changed', self.handle_focused_col_changed)
    signals.connect_signal(self._listbox, 'focused_row_changed', self.handle_focused_row_changed)

  def update_footer(self):
    # Update footer.
    focus_row, focus_col = self._listbox.focused_cell
    focus_str = '%d:%d' % (focus_row+1, focus_col)
    percent_str = '%0.01f%%' % (100.0 * focus_row / len(self._df))
    self._footer_header.base_widget.contents[1][0].base_widget.set_text(focus_str)
    self._footer_header.base_widget.contents[2][0].base_widget.set_text(percent_str)
    body_val = self._df.iloc[focus_row, focus_col]
    body_str = str(body_val) if pd.api.types.is_numeric_dtype(body_val) else json.dumps(body_val)
    column_str = '%s: ' % (self._df.columns[focus_col],)
    # self._footer_body.set_text('%s: %s' % (column_str, body_str))
    self._footer_body.set_text([('footer-dim', column_str), body_str])

  def handle_focused_col_changed(self, focused_col):
    self._header.set_focus_column(focused_col)
    header_contents = self._wrapped_widget.header.contents[0][0].base_widget.contents
    for i, (col_attr_map, _) in enumerate(header_contents):
      if i == focused_col:
        col_attr_map.set_attr_map({None: 'column-header-hilighted'})
      else:
        col_attr_map.set_attr_map({None: 'column-header'})
    self.update_footer()

  def handle_focused_row_changed(self, focused_row):
    self.update_footer()


def main(csv_in=None, has_header=True):
  if has_header:
    df = pd.read_csv(csv_in, dtype=str, na_values=None)
  else:
    df = pd.read_csv(csv_in, dtype=str, na_values=None, header=None)
    fmt = '[%%0%dd]' % (len(str(len(df.columns))),)
    df.columns = [fmt % (i,) for i in df.columns]
  if csv_in is sys.stdin:
    # Re-open a clean stdin for curses.
    f_tty = open('/dev/tty')
    os.dup2(f_tty.fileno(), sys.stdin.fileno())
    csv_path = '-'
  else:
    csv_path = csv_in.name
  base = DataFrameTable(df=df, csv_path=csv_path)
  main_loop = urwid.MainLoop(
    widget=base,
    palette=PALETTE,
    unhandled_input=unhandled_input,
    event_loop=urwid.AsyncioEventLoop(loop=asyncio.get_event_loop()),
  )
  main_loop.screen.set_mouse_tracking(enable=False)
  return main_loop.run()


if __name__ == '__main__':
  sys.exit(main(sys.argv))
