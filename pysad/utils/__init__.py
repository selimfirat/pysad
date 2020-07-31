
def _iterate(X):
    iterator = ArrayIterator(shuffle=False)
    for xi in iterator.iter(X):
        yield xi, None
