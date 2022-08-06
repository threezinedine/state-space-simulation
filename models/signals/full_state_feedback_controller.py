import numpy as np
from .signal import Signal
from ..timer import ITimer
from .i_feedback import IFeedBack


class FullStateFeedBackController(Signal):
    def __init__(self, timer:ITimer, feedback_model:IFeedBack):
        Signal.__init__(self, timer)
        self._feedback_model = feedback_model 

    def get_input(self):
        self._index += 1
        return self._feedback_model.get_feedback()
