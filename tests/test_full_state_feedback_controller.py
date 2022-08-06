import unittest
from parameterized import parameterized
from unittest.mock import Mock
from models.timer import ITimer
from models.signals import IFeedBack, FullStateFeedBackController
import numpy as np


DTYPE = np.float32


class TestFullStateFeedbackController(unittest.TestCase):
    @parameterized.expand([
            [np.array([[1., 1.]]), np.array([[2., 3.]]), np.array([[5.]]), DTYPE],
            [np.array([[1., 1.]]), None, np.array([[2.]]), DTYPE],
            [np.array([[2., 1.]]), None, np.array([[3.]]), DTYPE],
            [np.array([[2., 1.]]), None, np.array([[3.]]), np.float64] 
        ]) 
    def test_get_input_method(self, input_state, gain_param, expected, dtype):
        timer = Mock(spec=ITimer)

        feedback_model = Mock(spec=IFeedBack)
        feedback_model.get_feedback.return_value = input_state

        controller = FullStateFeedBackController(timer, feedback_model, gain_param)

        output = controller.get_input(dtype=dtype)

        assert isinstance(output, np.ndarray)
        assert output.dtype == dtype
        self.assertListEqual(output.tolist(), expected.tolist())
