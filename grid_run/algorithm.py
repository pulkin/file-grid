def eval_all(statements, names, builtins=None):
    """Evaluates all expressions from the dict"""
    if builtins is None:
        builtins = {}
    delta = set(statements).intersection(set(builtins))
    if len(delta) != 0:
        raise ValueError(f"statements overlap with builtins: {', '.join(map(repr, delta))}")
    delta = set(names).intersection(set(builtins))
    if len(delta) != 0:
        raise ValueError(f"names overlap with builtins: {', '.join(map(repr, delta))}")

    result = names.copy()
    results_and_builtins = {**result, **builtins}
    statements = statements.copy()

    while True:
        leave = True
        # iterate over programs to figure out which ones can be evaluated
        transaction = []
        for k, v in statements.items():
            if len(v.names_missing(results_and_builtins)) == 0:
                transaction.append(k)
                result[k] = results_and_builtins[k] = v.eval(results_and_builtins)
                leave = False
        if leave:
            if len(statements) > 0:
                info = []
                for v in statements.values():
                    info.append(f"{v}: missing {', '.join(map(repr, v.names_missing(results_and_builtins)))}")
                info = "\n".join(info)
                raise ValueError(
                    f"{len(statements)} expressions cannot be evaluated:\n{info}")
            return result
        else:
            for i in transaction:
                del statements[i]
