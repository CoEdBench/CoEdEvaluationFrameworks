import os
import pickle
import hashlib
from typing import List, Tuple

# --- 1. Import repo_ops module ---
# We need to directly manipulate this module's global variables
from src.infrastructure.locagent.plugins.location_tools.repo_ops import repo_ops

from src.infrastructure.locagent.dependency_graph.build_graph import (
    build_graph,
    NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS
)
from src.infrastructure.locagent.dependency_graph.traverse_graph import (
    RepoEntitySearcher, RepoDependencySearcher
)


class LocalCodeAnalyzer:
    def __init__(self, repo_path: str, cache_dir: str = "./graph_cache"):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Repository not found at: {self.repo_path}")

        # --- 2. Initialize graph database (with cache) ---
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        repo_hash = hashlib.md5(self.repo_path.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"graph_{repo_hash}.pkl")

        if os.path.exists(cache_file):
            print(f"📂 [LocalAnalyzer] Loading cached graph for {os.path.basename(repo_path)}...")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            print(f"🔄 [LocalAnalyzer] Building graph from scratch for {os.path.basename(repo_path)}...")
            self.graph = build_graph(self.repo_path, fuzzy_search=True, global_import=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)

        # Keep a local reference for _locate_specific_node
        self.entity_searcher = RepoEntitySearcher(self.graph)
        self.dep_searcher = RepoDependencySearcher(self.graph)

        # --- 3. [Key Step] Inject the graph into repo_ops' global state ---
        # So repo_ops.explore_tree_structure can access data when calling get_graph()
        print("💉 Injecting graph into repo_ops global state...")
        repo_ops.DP_GRAPH = self.graph
        repo_ops.DP_GRAPH_ENTITY_SEARCHER = self.entity_searcher
        repo_ops.DP_GRAPH_DEPENDENCY_SEARCHER = self.dep_searcher

        # If repo_ops has other needed global variables (e.g., CURRENT_ISSUE_ID), initialize as needed
        # repo_ops.CURRENT_ISSUE_ID = "local_analysis"

    def _get_node_details(self, node_id):
        """Get complete node information"""
        if not self.graph.has_node(node_id):
            return {"id": node_id, "error": "Node not found"}
        data = self.graph.nodes[node_id]
        return {
            "id": node_id,
            "type": data.get('type', 'unknown'),
            "file": node_id.split(':')[0],
            "start_line": data.get('start_line', -1),
            "end_line": data.get('end_line', -1),
        }

    def _locate_specific_node(self, file_path: str, line_num: int):
        """
        Recursive precise location: File -> Class -> Method
        Uses the local entity_searcher for fast location
        """
        if os.path.isabs(file_path):
            rel_file_path = os.path.relpath(file_path, self.repo_path)
        else:
            rel_file_path = file_path

        if not self.entity_searcher.has_node(rel_file_path):
            return None, None

        current_best_node_id = rel_file_path
        current_best_node_type = NODE_TYPE_FILE

        file_node_data = self.entity_searcher.get_node_data([rel_file_path])[0]
        current_range = float('inf')
        if 'start_line' in file_node_data and 'end_line' in file_node_data:
            current_range = file_node_data['end_line'] - file_node_data['start_line']

        search_queue = [rel_file_path]

        while search_queue:
            parent_id = search_queue.pop(0)
            child_ids, _ = self.dep_searcher.get_neighbors(
                parent_id,
                direction='forward',
                etype_filter=[EDGE_TYPE_CONTAINS]
            )

            if not child_ids:
                continue

            child_nodes = self.entity_searcher.get_node_data(child_ids)

            for node in child_nodes:
                if 'start_line' not in node or 'end_line' not in node:
                    continue

                s_line = node['start_line']
                e_line = node['end_line']

                if s_line <= line_num <= e_line:
                    node_range = e_line - s_line
                    if node_range < current_range:
                        current_range = node_range
                        current_best_node_id = node['node_id']
                        current_best_node_type = node['type']

                        if node['type'] in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]:
                            search_queue.append(node['node_id'])

        return current_best_node_id, current_best_node_type

    def get_co_edit_context(self, file_path: str, line_num: int):
        """
        [Core Entry] Get co-edit context
        """
        # 1. Precisely locate node
        focus_id, focus_type = self._locate_specific_node(file_path, line_num)

        if not focus_id:
            return f"❌ Error: Could not locate any node at {file_path}:{line_num}"

        print(f"📍 Focus Node: [{focus_type}] {focus_id}")
        node_details = self._get_node_details(focus_id)

        # 2. Directly call repo_ops methods
        # Since we already set DP_GRAPH in __init__, we can call directly without passing graph parameter

        # print("🔍 Querying Upstream Dependencies via repo_ops...")
        # upstream_report = repo_ops.explore_tree_structure(
        #     start_entities=[focus_id],
        #     direction='upstream',
        #     traversal_depth=2
        # )
        #
        # print("🔍 Querying Downstream Dependencies via repo_ops...")
        # downstream_report = repo_ops.explore_tree_structure(
        #     start_entities=[focus_id],
        #     direction='downstream',
        #     traversal_depth=2
        # )
        report = repo_ops.explore_tree_structure(
            start_entities=[focus_id],
            direction='both',
            traversal_depth=3
        )
        # 3. Format output
        result = []
        result.append(f"### 🎯 Focus Entity")
        result.append(f"- **ID**: `{focus_id}`")
        result.append(f"- **Type**: `{focus_type}`")
        result.append(f"- **Location**: Line {node_details['start_line']} - {node_details['end_line']}")
        result.append("")

        result.append(f"### Dependencies ")
        result.append(report if report else "(No  dependencies found)")
        result.append("")


        return "\n".join(result)
