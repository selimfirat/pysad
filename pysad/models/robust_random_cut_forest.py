from pysad.core.base_model import BaseModel


class RobustRandomCutForest(BaseModel):

    def __init__(self, num_trees=4, shingle_size=4, tree_size=256, **kwargs):
        import rrcf
        super().__init__(**kwargs)

        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees

        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)

        self.index = 0

    def fit_partial(self, X, y=None):

        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)

            tree.insert_point(X, index=self.index)

        self.index += 1

        return self

    def score_partial(self, X):
        leaf = rrcf.find_duplicate(X)

        score = 0.0
        for tree in self.forest:
            if leaf is None:
                tree.insert_point(X, index="test_point")
                score += 1.0 * tree.codisp("test_point") / self.num_trees
                tree.forget_point("test_point")
            else:
                score += 1.0 * tree.codisp(leaf) / self.num_trees

        return score
