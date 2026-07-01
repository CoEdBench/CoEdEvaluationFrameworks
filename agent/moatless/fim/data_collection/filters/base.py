from abc import ABC, abstractmethod
from pydriller import Commit

from moatless.fim.data_collection.config.settings import MiningConfig


class BaseFilter(ABC):
    @abstractmethod
    def check(self, commit: Commit, config: MiningConfig) -> bool:
        """Return True to keep the Commit, False to discard"""
        pass
