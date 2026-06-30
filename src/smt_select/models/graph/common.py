"""Shared graph conversion and vocabulary utilities for GIN selectors."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from smt_select.representations.graph_rep import smt_graph_to_gin

UNK_TYPE_INDEX = 0


class NodeVocabulary:
    """Map graph-node type strings to embedding indices."""

    def __init__(self) -> None:
        self._type_to_idx: dict[str, int] = {}
        self._frozen = False

    def add_type(self, type_name: str) -> int:
        if type_name in self._type_to_idx:
            return self._type_to_idx[type_name]
        if self._frozen:
            return UNK_TYPE_INDEX
        index = len(self._type_to_idx) + 1
        self._type_to_idx[type_name] = index
        return index

    def add_graph_dict(self, graph_dict: dict) -> None:
        for node in graph_dict.get("nodes", {}).values():
            if isinstance(node, dict) and "type" in node:
                self.add_type(node["type"])

    def get_index(self, type_name: str) -> int:
        return self._type_to_idx.get(type_name, UNK_TYPE_INDEX)

    def freeze(self) -> None:
        self._frozen = True

    def num_types(self) -> int:
        return len(self._type_to_idx)

    def type_names(self) -> list[str]:
        return [
            type_name
            for type_name, _ in sorted(
                self._type_to_idx.items(),
                key=lambda item: item[1],
            )
        ]


def build_vocabulary_from_graph_dicts(
    graph_dicts: list[dict],
) -> NodeVocabulary:
    vocabulary = NodeVocabulary()
    for graph_dict in graph_dicts:
        vocabulary.add_graph_dict(graph_dict)
    vocabulary.freeze()
    return vocabulary


def graph_dict_to_gin_data(
    graph_dict: dict,
    vocabulary: NodeVocabulary,
) -> Data | None:
    gin_graph = smt_graph_to_gin(graph_dict)
    type_indices = [
        vocabulary.get_index(node_type)
        for node_type in gin_graph["node_types"]
    ]
    node_features = torch.tensor(type_indices, dtype=torch.long).unsqueeze(1)
    num_nodes = node_features.size(0)
    edges = gin_graph["edges"]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if (
            edge_index.numel()
            and (edge_index.min() < 0 or edge_index.max() >= num_nodes)
        ):
            return None
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
