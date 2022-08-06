from abc import ABC, abstractmethod


class IFeedBack(ABC):
    @abstractmethod
    def get_feedback(self):
        raise NotImplementedError()
