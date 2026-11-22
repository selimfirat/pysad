"""Load an SMD machine stream and score it in time order."""

from pysad.models import LODA
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    data = Data()
    # SMD has 28 machines: machine-1-1 through machine-3-11
    # Use the first machine of the first group
    X, y = data.get_data("smd/machine-1-1.txt")
    print(f"Points: {len(X)}, features: {X.shape[1]}, anomalies: {int(y.sum())}")

    # Stream in original time order (shuffle=False)
    model = LODA(random_state=0)
    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        # In a real application, you would track scores and labels here
