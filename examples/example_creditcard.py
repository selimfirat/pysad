"""Load credit-card transactions and stream them in their original time order."""

from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("creditcard/creditcard.arff")
    print(f"Transactions: {len(X)}, features: {X.shape[1]}, frauds: {int(y.sum())}")

    # Equal timestamps keep their source order; Class is separate from the features.
    for transaction, label in ArrayStreamer().iter(X[:5], y[:5]):
        print(f"Time: {transaction[0]:.0f}, amount: {transaction[-1]:.2f}, fraud: {label}")
