from typing import Hashable


def resolve_dependency_tree(tree: dict[Hashable, set[Hashable]]) -> list[Hashable]:
    """
    Resolves the dependency tree and figures out the order
    in which statements are evaluated.

    Parameters
    ----------
    tree
        Dependency tree: dict keys depend on a set of
        other keys.

    Returns
    -------
    A list of statement names to evaluate.
    """
    tree = {k: set(v) for k, v in tree.items()}  # free-to-change copy
    visited = set()
    result = []

    while True:
        # sweep dependencies
        for name, depends_on in tree.items():
            depends_on.difference_update(visited)

        # remove empty dependencies
        transaction = tuple(
            name
            for name, depends_on in tree.items()
            if len(depends_on) == 0
        )
        for name in transaction:
            del tree[name]

        # add to result
        result.extend(transaction)
        visited.update(transaction)

        if len(tree) == 0:
            return result

        if len(transaction) == 0:
            info = []
            for name, depends_on in tree.items():
                info.append(f"{name}: missing {depends_on}")
            info = "\n".join(info)
            raise ValueError(f"{len(tree)} nodes are not resolved:\n{info}")


def eval_all(statements: list, names: dict):
    """Evaluates all expressions from the dict"""
    result = []
    names = names.copy()
    for statement in statements:
        v = statement.eval(names)
        names[statement.name] = v
        result.append(v)
    return result
