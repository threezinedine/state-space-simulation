import numpy as np
from .signal import Signal
from ..timer import ITimer
from .i_feedback import IFeedBack


DTYPE = np.float32


class FullStateFeedBackController(Signal):
    def __init__(self, timer:ITimer, feedback_model:IFeedBack, gain_param:np.ndarray=None):
        Signal.__init__(self, timer)
        self._feedback_model = feedback_model 

        self._gain_param = gain_param

    def get_input(self, dtype=DTYPE):
        self._index += 1
        state = self._feedback_model.get_feedback()

        if self._gain_param is None:
            self._gain_param = np.ones_like(state, dtype=dtype)
            
        result = np.dot(state.reshape(1, -1), self._gain_param.reshape(-1, 1))
        return result.astype(dtype)
