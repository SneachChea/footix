from typing import Any
from abc import abstractmethod, ABC


class Metric(ABC):
    higher_is_better: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.default_name: list[str] = []
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def add_state(self, name):
        if hasattr(self, name):
            raise ValueError("Name state already added")
        setattr(self, name, [])
        self.default_name.append(name)

    def reset(self):
        for name in self.default_name:
            setattr(self, name, [])
