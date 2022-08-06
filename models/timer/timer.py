from .i_timer import ITimer
import numpy as np


START_TIME = 0
STOP_TIME = 1
ITERVAL = .2


class Timer(ITimer):
    def __init__(self, start=START_TIME, stop=STOP_TIME, iterval=ITERVAL):
        self._start_time = start
        self._stop_time = stop
        self._iterval = iterval

    def get_length(self):
        return int((self._stop_time - self._start_time)/self._iterval)
