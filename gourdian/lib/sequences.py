import functools
import itertools
import math
import more_itertools
import numbers

from gourdian.lib import errors as lib_errors


class Sequence:
  def __init__(self, head, bomb, step=1):
    self.head = head
    self.bomb = bomb
    self.step = step

  def __repr__(self):
    return ('%s(head=%r, bomb=%r, step=%r)' %
            (self.__class__.__name__, self.head, self.bomb, self.step))

  def __len__(self):
    return math.ceil((self.bomb - self.head) / self.step)

  def __iter__(self):
    return itertools.chain(more_itertools.numeric_range(self.head, self.bomb, self.step))

  def __getitem__(self, index):
    # NOTE: Permits fetching the bomb index of this sequence, which isn't technically in the seq.
    if (index < 0) or (index > len(self)):
      raise IndexError('must be in (0, %r): %r' % (len(self), index))
    return self.head + (index * self.step)

  def __in__(self, value):
    return self.head <= value < self.bomb

  @property
  def shape(self):
    return (len(self),)

  @property
  def size(self):
    return len(self)

  def left_index_of(self, value, errors=lib_errors.RAISE):
    lib_errors.validate_errors(errors, coerce_ok=True, ignore_ok=True, raise_ok=True)
    if value < self.head:
      if errors == lib_errors.RAISE:
        raise ValueError('value too small (min=%r): %r' % (self.head, value,))
      if errors == lib_errors.COERCE:
        value = self.head
    return math.floor((value - self.head) / self.step)

  def right_index_of(self, value, errors=lib_errors.RAISE):
    lib_errors.validate_errors(errors, coerce_ok=True, ignore_ok=True, raise_ok=True)
    # NOTE: Permits fetching the bomb value of this sequence, which isn't technically in the seq.
    if value > self.bomb:
      if errors == lib_errors.RAISE:
        raise ValueError('value too large (max=%r): %r' % (self.bomb, value,))
      if errors == lib_errors.COERCE:
        value = self.bomb
    return math.ceil((value - self.head) / self.step)

  def to_js(self):
    return {
      'head': self.head,
      'bomb': self.bomb,
      'step': self.step,
    }


class MultiSequence:
  def __init__(self, sequences):
    self.sequences = tuple(sequences)

  def __repr__(self):
    return ('%s(sequences=%r)' % (self.__class__.__name__, self.sequences))

  def __len__(self):
    return functools.reduce(lambda acc, seq: acc * len(seq), self.sequences, 1)

  def __in__(self, values):
    return all((val in seq) for val, seq in zip(self.sequences, values))

  def __iter__(self):
    def helper(seq, rest, acc=()):
      if rest:
        for val in seq:
          yield from helper(seq=rest[0], rest=rest[1:], acc=(acc + (val,)))
      else:
        yield from ((acc + (val,),) for val in seq)
    return itertools.chain(*helper(seq=self.sequences[0], rest=self.sequences[1:]))

  def __getitem__(self, indices):
    if isinstance(indices, numbers.Number):
      # indices is a single int offset into this MultiSequence; convert to per-seq indices.
      indices = self.index_to_indices(index=indices)
    # [index, index, ...]: separately encodes the index for all sequences in self.sequences.
    if len(indices) != len(self.sequences):
      raise IndexError('expected indices of len=%d: %r' % (len(self.sequences), indices))
    ret = []
    for index, seq in zip(indices, self.sequences):
      ret.append(seq[index])
    return tuple(ret)

  @property
  def shape(self):
    return tuple(len(s) for s in self.sequences)

  @property
  def size(self):
    return len(self)

  def index_to_indices(self, index):
    if index < 0 or index >= len(self):
      raise IndexError('index out of bounds (len=%d): %r' % (len(self), index))
    stride = 1
    indices = []
    for seq in reversed(self.sequences):
      seq_index = math.floor(index/stride) % len(seq)
      stride = stride * len(seq)
      indices.insert(0, seq_index)
    return tuple(indices)

  def left_indices_of(self, values, errors=lib_errors.RAISE):
    return tuple(s.left_index_of(value=v, errors=errors) for s, v in zip(self.sequences, values))

  def to_js(self):
    return [s.to_js() for s in self.sequences]

  ##################################################################################################
  # SELECTORS
  def select(self, array, selector):
    for indices in selector:
      ret = []
      for head_index, bomb_index in indices:
        ret.extend(array[head_index:bomb_index])
      yield ret

  def get_sequence_selector(self, seq_index):
    sequences = self.sequences
    stops = functools.reduce(lambda acc, seq: acc * len(seq), sequences[:seq_index], 1)
    step = functools.reduce(lambda acc, seq: acc * len(seq), sequences[seq_index:], 1)
    stride = functools.reduce(lambda acc, seq: acc * len(seq), sequences[seq_index+1:], 1)
    for i in range(len(sequences[seq_index])):
      ret = []
      head_index = i * stride
      for j in range(stops):
        ret.append([head_index, head_index + stride])
        head_index += step
      yield ret
