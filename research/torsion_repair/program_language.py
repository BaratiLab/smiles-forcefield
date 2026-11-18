"""Canonical, order-independent finite language of verified repair programs.

No option letters. The same bounded proposal library is used by all policies.
"""
import itertools, json

PROGRAMS = [
    {
        "action": "revise" if scope == "broad" else "specialize",
        "scope": scope,
        "harmonics": list(range(1, h + 1)),
    }
    for scope in ("broad", "local")
    for h in (1, 2, 3, 4, 6)
]
ABSTAIN = {"action": "abstain", "scope": "broad", "harmonics": [1, 2, 3]}


def canonical(programs):
    if not programs or all(p["action"] in ("keep", "abstain") for p in programs):
        programs = [ABSTAIN]
    unique = {json.dumps(p, sort_keys=True, separators=(",", ":")): p for p in programs}
    return json.dumps(
        {"programs": [unique[k] for k in sorted(unique)]},
        sort_keys=True,
        separators=(",", ":"),
    )


def answers():
    return sorted(
        {canonical([ABSTAIN])}
        | {canonical([p]) for p in PROGRAMS}
        | {canonical(list(pair)) for pair in itertools.combinations(PROGRAMS, 2)}
    )


class TokenTrie:
    def __init__(self, sequences):
        self.root = {}
        for seq in sequences:
            node = self.root
            for token in seq:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if int(token) not in node:
                raise ValueError("Generated prefix left program language")
            node = node[int(token)]
        return sorted(node)


class Processor:
    def __init__(self, tok, prompt_length):
        self.trie = TokenTrie(
            [tok.encode(s + "<|im_end|>", add_special_tokens=False) for s in answers()]
        )
        self.prompt_length = prompt_length
        stop = tok.encode("<|im_end|>", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Single stop-token encoding required")
        self.stop = stop[0]

    def allowed(self, suffix):
        if self.stop in suffix:
            i = suffix.index(self.stop)
            if self.trie.allowed(suffix[:i]) != [self.stop] or any(
                t != self.stop for t in suffix[i:]
            ):
                raise ValueError("Invalid stop-token prefix")
            return [self.stop]  # MLX evaluates one token ahead before observing EOS.
        return self.trie.allowed(suffix)

    def __call__(self, tokens, logits):
        import mlx.core as mx

        allowed = self.allowed(tokens[self.prompt_length :].tolist())
        if not allowed:
            raise ValueError("Program exhausted without stop token")
        mask = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        mask[:, mx.array(allowed)] = 0
        return logits + mask
