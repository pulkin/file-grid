def linspace(start, end, steps):
    """Like numpy.linspace"""
    assert steps > 2
    assert isinstance(steps, int)
    result = []
    for i in range(steps):
        w = 1.0 * i / (steps - 1)
        result.append(start * (1 - w) + end * w)
    return result


builtins = {
    "range": range,
    "linspace": linspace,
}
