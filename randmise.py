def merge(plist, qlist, index=None):
    """Returns list of unique pairs of events and actors, none of which may on index"""
    if len(plist) == 0:
        return []
    if index is None:
        index = set()

    merged = None
    tried_pairs = set()
    for pvalue in plist:
        for i, qvalue in enumerate(qlist):
            pair = (pvalue, qvalue)
            if pair not in index and pair not in tried_pairs:
                new_index = index.union([pair])
                rest = merge(plist[1:], qlist[:i]+qlist[i+1:], new_index)

                if rest is not None:
                    # Found! Done.
                    merged = [pair] + rest
                    break
                else:
                    tried_pairs.add(pair)
        if merged is not None:
            break

    return merged