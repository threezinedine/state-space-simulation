from abc import ABC, abstractmethod


class IEvent(ABC):
    @abstractmethod
    def detect(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get(self) -> list:
        raise NotImplementedError()
