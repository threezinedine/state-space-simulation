from .i_timer import ITimer
import numpy as np


START_TIME = 0
STOP_TIME = 1
ITERVAL = .2
DTYPE = np.float32


class Timer(ITimer):
    def __init__(self, start=START_TIME, stop=STOP_TIME, iterval=ITERVAL, dtype=DTYPE):
        self._start_time = start
        self._stop_time = stop
        self._iterval = iterval
        self._dtype = dtype

    def get_length(self):
        return int((self._stop_time - self._start_time)/self._iterval)

    def get_time_input(self):
        length = self.get_length()
        output = np.zeros(shape=(length, ), dtype=DTYPE)
        tempt = self._start_time
        index = 0

        while tempt < self._stop_time and index < length:
            output[index] = tempt
            tempt += self._iterval
            index += 1

        return self._dtype(np.round_(output, decimals=3))
