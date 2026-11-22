"""Load a NASA SMAP or MSL telemetry channel and score it in time order."""

from pysad.models import LODA
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    data = Data()

    # SMAP has 54 channels (e.g., P-1, S-1, E-1 through A-9)
    X_smap, y_smap = data.get_data("smap/P-1.npy")
    print(
        f"SMAP P-1: points={len(X_smap)}, features={X_smap.shape[1]}, anomalies={int(y_smap.sum())}"
    )

    # MSL has 27 channels (e.g., M-1 through M-7, F-4 through T-13)
    X_msl, y_msl = data.get_data("msl/M-1.npy")
    print(f"MSL M-1: points={len(X_msl)}, features={X_msl.shape[1]}, anomalies={int(y_msl.sum())}")

    # Stream in original time order (shuffle=False)
    model = LODA(random_state=0)
    for x, label in ArrayStreamer().iter(X_smap, y_smap):
        score = model.fit_score_partial(x)
        # In a real application, you would track scores and labels here
