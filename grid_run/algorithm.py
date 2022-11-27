def eval_sort(statements: dict, names: set):
    """Figures out the order in which statements should be evaluated"""
    names = set(names)
    dependencies = {k: v.required for k, v in statements.items()}
    result = []
    delta = set(dependencies).intersection(names)
    if len(delta) != 0:
        raise ValueError(f"names and statements overlap: {', '.join(delta)}")

    while True:
        transaction = []
        for name, depends_on in dependencies.items():
            if len(depends_on - names) == 0:
                transaction.append(name)
                names.add(name)
                result.append(statements[name])
        if len(transaction) == 0:
            if len(dependencies) > 0:
                info = []
                for k, v in dependencies.items():
                    info.append(f"{statements[k]}: missing {', '.join(map(repr, set(v) - names))}")
                info = "\n".join(info)
                raise ValueError(
                    f"{len(dependencies)} expressions cannot be evaluated:\n{info}")
            return result
        else:
            for i in transaction:
                del dependencies[i]


def eval_all(statements: list, names: dict):
    """Evaluates all expressions from the dict"""
    result = []
    names = names.copy()
    for statement in statements:
        v = statement.eval(names)
        names[statement.name] = v
        result.append(v)
    return result
