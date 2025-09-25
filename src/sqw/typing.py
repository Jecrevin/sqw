import numpy as np

type Array1D[_DType: np.generic] = np.ndarray[tuple[int], np.dtype[_DType]]
