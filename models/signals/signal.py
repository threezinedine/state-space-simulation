from .i_signal import ISignal
from ..timer import ITimer


class Signal(ISignal):
    def __init__(self, timer:ITimer):
        self._timer = timer
        self._index = 0

    def is_finished(self) -> bool: 
        if self._index == self._timer.get_length():
            return True
        else:
            return False

