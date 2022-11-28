from pathlib import Path
from functools import partial

from .template import Template


def match_files(patterns, root=".", apply=None, exclude=None, recursive=False, hidden=False, allow_empty=False):
    if len(patterns) == 0 and not allow_empty:
        raise ValueError(f"no patterns provided")
    root = Path(root)
    if exclude is None:
        exclude = set()

    result = []
    matched_so_far = set()
    for pattern in patterns:
        anything_matched = False
        matched_total = 0
        matched_files = 0
        for match in (root.rglob if recursive else root.glob)(pattern):
            matched_total += 1
            if match.is_file() and match not in matched_so_far and (hidden or not match.name.startswith(".")):
                matched_so_far.add(match)
                matched_files += 1
                if match not in exclude:
                    if apply is not None:
                        match = apply(match)
                        if match is not None:
                            result.append(match)
                            anything_matched = True
                    else:
                        result.append(match)
                        anything_matched = True

        if not anything_matched:
            raise ValueError(f"pattern '{pattern}' in '{str(root)}' matched 0 files (matched total: {matched_total}, "
                             f"files: {matched_files})")
    return result


def _maybe_template(candidate):
    with open(candidate, 'r') as f:
        result = Template.from_file(f)
    if result.is_trivial():
        return
    return result


match_template_files = partial(match_files, apply=_maybe_template)
