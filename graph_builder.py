import inflect
import networkx as nx
from collections import defaultdict, deque



def _normalize_reasons(reasons):

    """
    turns reasons into list form for edge processing
    """
    if not reasons:
        return []
    if isinstance(reasons, str):
        return [reasons]
    return list(reasons)

def get_grammar(word, mcdi_info):
    """
    retrieves manually specified grammatical classiciation of a word from the MCDI info dict
    """
    entry = mcdi_info.get(word)
    if entry:
        return entry.get("grammar")
    return None


def infer_gramm(reasons):
    """Infer grammatical class from transform reasons."""
    reason_set = set(reasons) if reasons else set()
    if "plural possessive" in reason_set or "dumb plural possessive" in reason_set:
        return "plural_possessive_noun"
    if "plural" in reason_set or "dumb plural" in reason_set:
        return "plural_noun"
    if "possessive" in reason_set:
        return "singular_possessive_noun"
    return "singular_noun"

# EDGE HIERARCHY

TRANSFORM_ORDER = {
    "base": 0,
    "singular": 1,
    "plural": 2,
    "dumb plural": 2,
    "possessive": 3,
    "plural possessive": 4,
    "dumb plural possessive": 4,
    "compound": 5,
}

_PARENT_TRANSFORM = {
    "possessive":             "singular",
    "plural possessive":      "plural",
    "dumb plural possessive": "dumb plural",
}


def _get_all_edges(base, alt, reasons, all_alts):

    reasons = _normalize_reasons(reasons)
    reason_set = set(reasons)

    edges = []

    # compounds
    if "compound-childes_friendly" in reason_set:
        parent_reasons = [r for r in reasons if r != "compound-childes_friendly"]
        parent = base

        for a, m in all_alts.items():
            if a == alt:
                continue
            a_reasons = set(_normalize_reasons(m.get("reason", [])))
            if set(parent_reasons) == a_reasons:
                parent = a
                break

        return [(parent, "compound")]

    # gramm transform
    for r in reasons:

        if r == "compound-childes_friendly":
            continue

        if r in _PARENT_TRANSFORM:
            needed_parent = _PARENT_TRANSFORM[r]
            parent = base
            for a, m in all_alts.items():
                if a == alt:
                    continue
                if needed_parent in _normalize_reasons(m.get("reason", [])):
                    parent = a
                    break
            edges.append((parent, r))
            continue

        edges.append((base, r))

    # fallback if fail 
    if edges:
        return edges

    # last resort only
    return [(base, "base")]






# BUILD GRAPH 

def build_word_graph(base, alt_forms_dict, childes_counts, mcdi_info,
                     skip_compounds=False):
    """
    directed multigraph for mcdi word, alt forms, and corresponding data

    Node attributes:
      gramm: grammatical classification inferred from transform reasons
      category: MCDI category (inherited from base for alts)
      r_count: raw count in CHILDES for this exact word form
      prod: MCDI production score (inherited from base for alts)
      is_base: bool, True only for the root node
      reasons: list of transform reasons (empty list for base)
      initials: contributor initials from manual inclusion (None if N/A)
      source: source tag from manual inclusion (None if N/A)

    Edge attributes:
      transform - name of the transform that generated the child from the parent

    :param base: str - key in alt_forms_dict (original MCDI notation)
    :param alt_forms_dict: dict {base: {alt: {reason, initials, source}}}
    :param childes_counts: Counter - lowercased word -> CHILDES occurrence count
    :param mcdi_info: dict {base: {category, prod}}
    :param skip_compounds: bool - exclude compound-childes_friendly forms
    :return: nx.MultiDiGraph
    """
    G = nx.MultiDiGraph()
    alts = alt_forms_dict.get(base, {})
    info = mcdi_info.get(base, {"category": "unknown", "prod": None})

    # root
    G.add_node(
        base,
        gramm=get_grammar(base, mcdi_info) or "singular_noun",
        category=info.get("category", "unknown"),
        r_count=childes_counts.get(base, 0),
        prod=info.get("prod"),
        is_base=True,
        reasons=[],
        initials=None,
        source=None,
    )

    # alt filtering
    filtered_alts = {}
    for alt, meta in alts.items():
        if alt == base:
            continue
        reasons = _normalize_reasons(meta.get("reason", []))
        if skip_compounds and "compound-childes_friendly" in reasons:
            continue
        filtered_alts[alt] = {**meta, "reason": reasons}

    # add alt WITH PROVENANCE
    for alt, meta in filtered_alts.items():
        reasons = _normalize_reasons(meta.get("reason", []))
        G.add_node(
            alt,
            gramm=infer_gramm(reasons),
            category=info.get("category", "unknown"),
            r_count=childes_counts.get(alt, 0),
            prod=info.get("prod"),
            is_base=False,
            reasons=reasons,
            initials=meta.get("initials"),
            source=meta.get("source"),
        )

    # add edges with digraph enabled
    for alt, meta in filtered_alts.items():
        reasons = _normalize_reasons(meta.get("reason", []))
        for parent, edge_label in _get_all_edges(base, alt, reasons, filtered_alts):
            if parent not in G.nodes:
                parent = base
            G.add_edge(parent, alt, transform=edge_label)

    return G


# BFS layout for hierarchy 
def hierarchical_layout(G, root):
    """
    :param G: nx.MultiDiGraph
    :param root: root node
    :return: dict {node: (x, y)}
    """
    pos = {}
    level_nodes = defaultdict(list)
    visited = set()
    queue = deque([(root, 0)])
    visited.add(root)

    while queue:
        node, level = queue.popleft()
        level_nodes[level].append(node)
        for child in G.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append((child, level + 1))

    # anything abandoned goes here i guess... we'll fix this later
    # TODO: figure out wtf is going on with this
    for node in G.nodes:
        if node not in visited:
            max_lv = max(level_nodes.keys(), default=0) + 1
            level_nodes[max_lv].append(node)

    max_level = max(level_nodes.keys(), default=0)
    for level, nodes in level_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * 2.5
            y = max_level - level
            pos[node] = (x, y)

    return pos

# get MCDI info for graph building 
def build_mcdi_info(mcdi_ibi_df, base_col="base",
                    cat_col="category",
                    prod_col="sample_prod_col",
                    gram_col="grammar"):
    """
    Build a lookup dict from the preprocessed MCDI df.

    :return: dict {base_word: {category: str, prod: float|None, grammar: str|None}}
    """
    mcdi_info = {}

    for _, row in mcdi_ibi_df.drop_duplicates(subset=[base_col]).iterrows():
        key = row[base_col]

        prod = row.get(prod_col)
        grammar = row.get(gram_col)

        mcdi_info[key] = {
            "category": row.get(cat_col, "unknown"),
            "prod": float(prod) if prod is not None and str(prod) != "nan" else None,
            "grammar": None if grammar is None or str(grammar) == "nan" else grammar,
        }

    return mcdi_info