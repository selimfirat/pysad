from __future__ import annotations

import numbers
from collections.abc import Iterator

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class _Node:
    """One node of the fixed-depth Drain tree.

    A node at depth ``k`` is reached by the first ``k`` tokens of a log line. Its children are keyed
    by the next token, so each path down the tree spells one prefix of tokens. A missing token
    (the line is shorter than the tree depth) is keyed by ``None``. Leaf nodes sit at depth
    ``depth`` and hold the log templates whose lines share that path. When ``max_child`` stops a
    node from growing, a new template may be stored at that node instead.
    """

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.children: dict[str | None, _Node] = {}
        self.clusters: list[_LogCluster] = []


class _LogCluster:
    """One template and the log lines that parse to it.

    Args:
        template: The token sequence of the template, with ``"*"`` where templates differ.
        event_id: The event ID handed out for this template, in creation order.
    """

    def __init__(self, template: list[str], event_id: int) -> None:
        self.template = template
        self.event_id = event_id
        self.size = 1


class DrainLogParser(BaseTransformer):
    """Parses each log line into an event ID with Drain's fixed-depth tree :cite:`he2017drain`.

    Drain splits a log line into whitespace-separated tokens and walks one path down a tree whose
    depth is fixed. At depth ``k`` the path follows the ``k``-th token. A leaf holds the templates
    whose lines share that path. The new line is compared only with the templates in the leaf it
    reaches. The depth of the tree and the number of children of a node are bounded by ``depth``
    and ``max_child``, but a leaf can hold many templates, so the cost of one line grows with the
    number of templates in its leaf.

    A line matches a template when the share of agreeing tokens is at least ``st``. A ``"*"`` in a
    template matches any token. On a match, the line takes the template's event ID and the template
    is generalized where the line differs. On no match, a new template and a new event ID are
    created. Event IDs are handed out in the order templates first appear, so the IDs follow file
    order.

    ``fit_partial`` parses a line and remembers its template. ``transform_partial`` parses without
    remembering and returns the matched event ID as a float, or ``-1.0`` when no template matches.
    ``fit_transform_partial`` does both and returns the event ID.

    Deviations from the paper:

    * The paper checks templates only at the leaf. When ``max_child`` stops a node from growing, the
      new template is stored at that node, and the search returns the shallowest match along the
      path. This keeps the tree bounded and only matters when a node would exceed ``max_child``
      children.
    * Tokens are split on whitespace only. The optional regular expressions of the paper for masking
      dynamic parameters are not implemented.

    Args:
        depth (int): Fixed depth of the tree. It must be at least 1 (Default=4).
        st (float): Similarity threshold between 0 and 1. A line matches a template when at least
            this share of tokens agrees (Default=0.4).
        max_child (int): Largest number of child nodes a node may hold. It bounds the tree
            (Default=100).

    Raises:
        TypeError: If ``depth`` or ``max_child`` is not an int, or ``st`` is not a real number.
        ValueError: If ``depth`` or ``max_child`` is less than 1, or ``st`` is outside [0, 1].
    """

    depth: int
    st: float
    max_child: int
    root_node: _Node
    _next_id: int

    def __init__(self, depth: int = 4, st: float = 0.4, max_child: int = 100) -> None:
        super().__init__(1)

        if isinstance(depth, bool) or not isinstance(depth, numbers.Integral):
            raise TypeError(f"depth must be an int, got {depth!r}.")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        if isinstance(st, bool) or not isinstance(st, numbers.Real):
            raise TypeError(f"st must be a real number, got {st!r}.")
        st = float(st)
        if not 0.0 <= st <= 1.0:
            raise ValueError(f"st must be between 0 and 1, got {st}.")
        if isinstance(max_child, bool) or not isinstance(max_child, numbers.Integral):
            raise TypeError(f"max_child must be an int, got {max_child!r}.")
        if max_child < 1:
            raise ValueError(f"max_child must be at least 1, got {max_child}.")

        self.depth = int(depth)
        self.st = st
        self.max_child = int(max_child)
        self.root_node = _Node(0)
        self._next_id = 1

    def _line(self, X: np.ndarray | str) -> str:
        """Returns the single log line held by ``X``.

        Args:
            X (np.ndarray | str): A log line as a string, or a 0-D or 1-D object array holding one
                string.

        Returns:
            str: The log line.

        Raises:
            ValueError: If ``X`` does not hold exactly one string.
            TypeError: If the held value is not a string.
        """
        if isinstance(X, str):
            return X
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 0:
            value = arr.item()
        elif arr.ndim == 1 and arr.size == 1:
            value = arr[0]
        else:
            raise ValueError(
                f"DrainLogParser expected a single log line, got an input of shape {np.shape(X)}."
            )
        if not isinstance(value, str):
            raise TypeError(
                f"DrainLogParser expected a log line string, got {type(value).__name__}."
            )
        return value

    def _search(self, tokens: list[str]) -> _LogCluster | None:
        """Returns the best-matching template along the path of ``tokens``.

        The search walks one path down the fixed-depth tree and checks the templates of every node
        on that path, so it never touches templates stored elsewhere. It returns the shallowest
        match whose similarity is at least ``st``, or None when no template matches.

        Args:
            tokens (list[str]): The whitespace-separated tokens of a log line.

        Returns:
            _LogCluster | None: The best matching cluster, or None when there is no match.
        """
        node: _Node | None = self.root_node
        for level in range(self.depth + 1):
            if node is None:
                return None
            match = _best_match(node.clusters, tokens, self.st)
            if match is not None:
                return match
            if level == self.depth:
                break
            token = tokens[level] if level < len(tokens) else None
            node = node.children.get(token)
        return None

    def _insert(self, cluster: _LogCluster, tokens: list[str]) -> None:
        """Stores ``cluster`` at the leaf reached by ``tokens``.

        If ``max_child`` stops the path from growing, the cluster is stored at the deepest node
        that still accepts a child, which keeps the tree bounded.

        Args:
            cluster (_LogCluster): The new template cluster.
            tokens (list[str]): The whitespace-separated tokens of a log line.
        """
        node = self.root_node
        for level in range(self.depth):
            token = tokens[level] if level < len(tokens) else None
            child = node.children.get(token)
            if child is None:
                if len(node.children) >= self.max_child:
                    node.clusters.append(cluster)
                    return
                child = _Node(level + 1)
                node.children[token] = child
            node = child
        node.clusters.append(cluster)

    def _find_or_create(self, tokens: list[str]) -> _LogCluster:
        """Returns the matching cluster for ``tokens`` or creates and stores a new one."""
        cluster = self._search(tokens)
        if cluster is None:
            cluster = _LogCluster(list(tokens), self._next_id)
            self._next_id += 1
            self._insert(cluster, tokens)
        else:
            cluster.size += 1
            cluster.template = _merge(cluster.template, tokens)
        return cluster

    def fit_partial(self, X: np.ndarray | str) -> DrainLogParser:
        """Parses the next log line and remembers its template.

        Args:
            X (np.ndarray | str): The next log line as a string, or a 0-D or 1-D object array
                holding one string.

        Returns:
            object: The parser.
        """
        self._find_or_create(self._line(X).split())
        return self

    def transform_partial(self, X: np.ndarray | str) -> np.ndarray:
        """Returns the event ID of the next log line without changing the parser.

        Args:
            X (np.ndarray | str): The next log line as a string, or a 0-D or 1-D object array
                holding one string.

        Returns:
            np.float64 array of shape (1,): The event ID of the matched template, or ``-1.0`` when
                no template matches.
        """
        cluster = self._search(self._line(X).split())
        if cluster is None:
            return np.array([-1.0], dtype=np.float64)
        return np.array([float(cluster.event_id)], dtype=np.float64)

    def fit_transform_partial(self, X: np.ndarray | str) -> np.ndarray:
        """Parses the next log line, remembers its template, and returns its event ID.

        Args:
            X (np.ndarray | str): The next log line as a string, or a 0-D or 1-D object array
                holding one string.

        Returns:
            np.float64 array of shape (1,): The event ID of the line.
        """
        cluster = self._find_or_create(self._line(X).split())
        return np.array([float(cluster.event_id)], dtype=np.float64)

    def _lines(self, X: np.ndarray | str) -> Iterator[str]:
        """Yields each log line held by ``X`` as a string.

        Args:
            X (np.ndarray | str): Log lines as a 1-D array of any dtype, or a single log line as a
                string.

        Returns:
            Iterator[str]: The log lines in order.

        Raises:
            TypeError: If an element of ``X`` is not a string. The message and the error type match
                the ones of the single-instance path, so both paths reject the same inputs.
        """
        if isinstance(X, str):
            yield X
            return
        for element in np.asarray(X, dtype=object).ravel():
            yield self._line(element)

    def fit(self, X: np.ndarray | str) -> DrainLogParser:
        """Parses every log line of ``X`` in order and remembers its templates.

        Args:
            X (np.ndarray | str): Log lines as a 1-D object array of strings, or a single log line
                as a string.

        Returns:
            object: The parser.
        """
        for line in self._lines(X):
            self.fit_partial(np.array(line, dtype=object))
        return self

    def transform(self, X: np.ndarray | str) -> np.ndarray:
        """Returns the event ID of every log line of ``X`` without changing the parser.

        Args:
            X (np.ndarray | str): Log lines as a 1-D object array of strings, or a single log line
                as a string.

        Returns:
            np.float64 array of shape (num_instances, 1): The event ID of each line, or ``-1.0``
                when no template matches.
        """
        values = [self.transform_partial(np.array(line, dtype=object)) for line in self._lines(X)]
        if not values:
            return np.empty((0, 1), dtype=np.float64)
        return np.array(values, dtype=np.float64)

    def fit_transform(self, X: np.ndarray | str) -> np.ndarray:
        """Parses every log line of ``X`` in order and returns its event IDs.

        Args:
            X (np.ndarray | str): Log lines as a 1-D object array of strings, or a single log line
                as a string.

        Returns:
            np.float64 array of shape (num_instances, 1): The event ID of each line.
        """
        values = [
            self.fit_transform_partial(np.array(line, dtype=object)) for line in self._lines(X)
        ]
        if not values:
            return np.empty((0, 1), dtype=np.float64)
        return np.array(values, dtype=np.float64)


def _best_match(clusters: list[_LogCluster], tokens: list[str], st: float) -> _LogCluster | None:
    """Returns the most similar cluster whose similarity is at least ``st``.

    Ties keep the first cluster. Returns None when no cluster reaches the threshold.

    Args:
        clusters (list[_LogCluster]): The templates of one node.
        tokens (list[str]): The tokens of a log line.
        st (float): The similarity threshold.

    Returns:
        _LogCluster | None: The best matching cluster, or None.
    """
    best: _LogCluster | None = None
    best_sim = st
    for cluster in clusters:
        sim = _similarity(cluster.template, tokens)
        if sim < st:
            continue
        if best is None or sim > best_sim:
            best = cluster
            best_sim = sim
    return best


def _similarity(template: list[str], tokens: list[str]) -> float:
    """Returns the share of agreeing tokens between a template and a log line.

    A ``"*"`` in the template matches any token. Positions present in only one sequence do not
    agree, so length differences lower the similarity.

    Args:
        template (list[str]): The template tokens.
        tokens (list[str]): The log line tokens.

    Returns:
        float: The similarity in [0, 1].
    """
    n = max(len(template), len(tokens))
    if n == 0:
        return 1.0
    matches = 0
    for i in range(n):
        t = template[i] if i < len(template) else None
        tok = tokens[i] if i < len(tokens) else None
        if t is None or tok is None:
            continue
        if t == "*" or t == tok:
            matches += 1
    return matches / n


def _merge(template: list[str], tokens: list[str]) -> list[str]:
    """Generalizes ``template`` with ``tokens``.

    Kept tokens are written where the two sequences agree, and ``"*"`` is written where they differ.
    Tokens present in only one sequence are kept.

    Args:
        template (list[str]): The current template tokens.
        tokens (list[str]): The new log line tokens.

    Returns:
        list[str]: The generalized template.
    """
    n = max(len(template), len(tokens))
    out: list[str] = []
    for i in range(n):
        t = template[i] if i < len(template) else None
        tok = tokens[i] if i < len(tokens) else None
        if t is None:
            # ``i`` is below the maximum length, so the line still has a token here.
            out.append(tok)  # type: ignore[arg-type]
        elif tok is None:
            out.append(t)
        elif t == "*" or t == tok:
            out.append(t)
        else:
            out.append("*")
    return out
