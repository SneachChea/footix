from typing import Any, Protocol


class ProtoModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...
    def predict(self, HomeTeam: str, AwayTeam: str) -> Any:
        ...