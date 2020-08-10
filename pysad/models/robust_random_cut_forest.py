from pysad.core.base_model import BaseModel
from rrcf import rrcf


class RobustRandomCutForest(BaseModel):
    """Robust Random Cut Forest model :cite:`guha2016robust`. The implementation uses `rrcf library <https://github.com/kLabUM/rrcf>`_ :cite:`bartos_2019_rrcf`.

        Args:
            num_trees: int
                The number of trees.
            shingle_size: int (Default=4)
                The shingle size.
            tree_size: The tree size
                The tree size.
    """

    def __init__(self, num_trees=4, shingle_size=4, tree_size=256):

        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees

        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)

        self.index = 0

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                Ignored since the model is unsupervised.

        Returns:
            self: object
                Returns the self.
        """
        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)

            tree.insert_point(X, index=self.index)

        self.index += 1

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score: float
                The anomalousness score of the input instance.
        """

        score = 0.0
        for tree in self.forest:
            leaf = tree.find_duplicate(X)
            if leaf is None:
                tree.insert_point(X, index="test_point")
                score += 1.0 * tree.codisp("test_point") / self.num_trees
                tree.forget_point("test_point")
            else:
                score += 1.0 * tree.codisp(leaf) / self.num_trees

        return score
