from itertools import product


def combinations(dict_of_iterables):
    keys = dict_of_iterables.keys()
    for i in product(*dict_of_iterables.values()):
        yield dict(zip(keys, i))
