"""Group frames or tiles connected by reliable pairwise measurements."""


def connected_components(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda group: group[0])
