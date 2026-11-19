from __future__ import annotations

import numbers
from collections.abc import Iterator

import numpy as np

from pysad.core.base_transformer import BaseTransformer

_WILDCARD = "*"


class _Node:
    """One node of the fixed-depth tree.

    The root sits at level 0. Its children are the length nodes, and each length node is keyed by
    the number of tokens of a log line. The children of a length node are the token nodes, and each
    token node is keyed by the token at that position. A token that holds a digit uses the key
    ``"*"``, so all such tokens share one node. A missing token uses the key ``None``. Leaf nodes
    hold the log templates whose lines share that path. When ``max_child`` stops a node from
    growing, a new template may be stored at that node instead.
    """

    def __init__(self, level: int) -> None:
        self.level = level
        self.children: dict[str | int | None, _Node] = {}
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

    Drain splits a log line into whitespace-separated tokens. The tree has one length layer and
    ``depth - 2`` token layers, so the leaf sits at level ``depth - 1``. The root holds one child
    per token count, so lines of different lengths never meet. Each length node holds the token
    nodes, and each token node is keyed by the token at that position. A token that holds a digit
    uses the key ``"*"``, so all such tokens share one node and the tree does not grow a branch
    for every value. A leaf holds the templates whose lines share that path. The parser compares
    the new line only with the templates in the leaf it reaches. The depth of the tree and the
    number of children of a node are bounded by ``depth`` and ``max_child``, but a leaf can hold
    many templates, so the cost of one line grows with the number of templates in its leaf.

    A line matches a template when the share of agreeing tokens is at least ``st``. A ``"*"`` in a
    template matches any token. On a match, the line takes the event ID of the template, and the
    template is generalized where the line differs. On no match, the parser creates a new template
    and a new event ID. Event IDs are handed out in the order the templates first appear, so the
    IDs follow file order.

    ``fit_partial`` parses a line and remembers its template. ``transform_partial`` parses without
    remembering and returns the event ID of the matched template as a float, or ``-1.0`` when no
    template matches. ``fit_transform_partial`` does both and returns the event ID.

    Differences from the paper and its reference implementations:

    * The paper compares the templates only at the leaf. When ``max_child`` stops a node from
      growing, the parser stores the new template at that node. The parser then compares the
      templates of every node on the path and returns the shallowest match. This keeps the tree
      bounded. It only matters when a node would exceed ``max_child`` children.
    * The paper lets the user mask dynamic parameters with regular expressions. The parser splits
      the line on whitespace only and does not mask parameters with regular expressions.
    * The paper does not say how to break a tie between two templates with the same similarity.
      The parser keeps the first template. The reference implementations keep the template with
      more parameters.
    * A ``"*"`` in a template matches any token and counts toward the similarity. The reference
      implementations skip the wildcard positions of a template.

    Args:
        depth (int): Fixed depth of the tree. It must be at least 2. The tree has one length layer
            and ``depth - 2`` token layers (Default=4).
        st (float): Similarity threshold between 0 and 1. A line matches a template when at least
            this share of tokens agrees (Default=0.4).
        max_child (int): Largest number of child nodes a node may hold. It bounds the tree
            (Default=100).

    Raises:
        TypeError: If ``depth`` or ``max_child`` is not an int, or ``st`` is not a real number.
        ValueError: If ``depth`` is less than 2, ``max_child`` is less than 1, or ``st`` is outside
            [0, 1].
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
        if depth < 2:
            raise ValueError(f"depth must be at least 2, got {depth}.")
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

    def _path(self, tokens: list[str]) -> Iterator[_Node]:
        """Yields the nodes on the path of ``tokens``, from the root to the leaf.

        The path starts at the root and follows the length node of ``tokens``. It then follows one
        token node per token layer. The path ends at the first node that does not exist.

        Args:
            tokens (list[str]): The whitespace-separated tokens of a log line.

        Yields:
            _Node: The nodes on the path, from the root to the leaf.
        """
        node = self.root_node
        yield node
        child = node.children.get(len(tokens))
        if child is None:
            return
        node = child
        yield node
        for level in range(self.depth - 2):
            key = _token_key(tokens[level] if level < len(tokens) else None)
            child = node.children.get(key)
            if child is None:
                return
            node = child
            yield node

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
        for node in self._path(tokens):
            match = _best_match(node.clusters, tokens, self.st)
            if match is not None:
                return match
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
        length = len(tokens)
        child = node.children.get(length)
        if child is None:
            if len(node.children) >= self.max_child:
                node.clusters.append(cluster)
                return
            child = _Node(1)
            node.children[length] = child
        node = child
        for level in range(self.depth - 2):
            key = _token_key(tokens[level] if level < len(tokens) else None)
            child = node.children.get(key)
            if child is None:
                if len(node.children) >= self.max_child:
                    node.clusters.append(cluster)
                    return
                child = _Node(level + 2)
                node.children[key] = child
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


def _token_key(token: str | None) -> str | None:
    """Returns the key of ``token`` in the tree.

    A token that holds a digit uses the key ``"*"``, so all such tokens share one node. A missing
    token uses the key ``None``.

    Args:
        token (str | None): The token at one position of a log line, or None when the line has no
            token at that position.

    Returns:
        str | None: The key of the token in the tree.
    """
    if token is None:
        return None
    if any(char.isdigit() for char in token):
        return _WILDCARD
    return token


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
        if t == _WILDCARD or t == tok:
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
        elif t == _WILDCARD or t == tok:
            out.append(t)
        else:
            out.append(_WILDCARD)
    return out
