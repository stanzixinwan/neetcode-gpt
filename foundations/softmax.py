import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        softmax = []
        sub_z = []
        max_z = np.max(z)
        for i in range(len(z)):
            s = z[i] - max_z
            sub_z.append(np.exp(s))
        denominator = np.sum(sub_z)
        for i in range(len(z)):
            sm = sub_z[i] / denominator
            softmax.append(np.round(sm, 4))
        return softmax
