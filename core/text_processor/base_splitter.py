from typing import List
from abc import ABC, abstractmethod


class BaseTextProcessor(ABC):
    @abstractmethod
    def split_text(self, texts: List[str]):
        pass
