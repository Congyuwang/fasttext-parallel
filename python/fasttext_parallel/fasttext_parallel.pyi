__all__ = ["load_model", "FastText"]

from typing import Tuple, List, Dict
import numpy as np

def load_model(path: str) -> FastText: ...

class FastText:
    def batch(self, texts: List[str], label_to_int: Dict[str, int], k: int = 1, threshold: float = -1.0) -> Tuple[np.ndarray, np.ndarray]: ...
