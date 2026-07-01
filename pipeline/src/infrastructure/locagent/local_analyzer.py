import os
import pickle
import hashlib
import shutil
import stat
from itertools import takewhile
from typing import List, Tuple
import logging
logger = logging.getLogger(__name__)

# --- Project imports ---

# Assume dependency_graph is in the current directory
from src.infrastructure.locagent.dependency_graph.build_graph import (
    build_graph,
    NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS, EDGE_TYPE_INHERITS
)
from src.infrastructure.locagent.dependency_graph.traverse_graph import (
    RepoEntitySearcher, RepoDependencySearcher
)

def remove_readonly(func, path, excinfo):
    """Handle the issue where shutil.rmtree cannot delete read-only files on Windows"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


class LocalCodeAnalyzer:
    def __init__(self, repo_path: str, cache_dir: str = "./graph_cache"):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Repository not found at: {self.repo_path}")

        # --- 1. Initialize graph database (with cache) ---
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Use path hash as cache filename to avoid collisions
        repo_hash = hashlib.md5(self.repo_path.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"graph_{repo_hash}.pkl")

        if os.path.exists(cache_file):
            # os.remove(cache_file)
            print(f"📂 [LocalAnalyzer] Loading cached graph for {os.path.basename(repo_path)}...")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            print(f"🔄 [LocalAnalyzer] Building graph from scratch for {os.path.basename(repo_path)}...")
            self.graph = build_graph(self.repo_path, fuzzy_search=True, global_import=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)

        self.entity_searcher = RepoEntitySearcher(self.graph)
        self.dep_searcher = RepoDependencySearcher(self.graph)

    def _get_node_details(self, node_id, add_line_numbers=False):
        """
        Get complete node information, including source code
        Args:
            node_id: Node ID
            add_line_numbers: Whether to prepend line numbers to source code (e.g., "   10: def foo():")
        """
        if not self.graph.has_node(node_id):
            return {"id": node_id, "error": "Node not found"}

        data = self.graph.nodes[node_id]
        code = data.get('code', '')
        start_line = data.get('start_line', 1)  # Default to 1 in case no line number info

        # If line numbers need to be added
        if add_line_numbers and code:
            lines = code.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines):
                current_line = start_line + i
                # Format as "   10: code..."
                formatted_lines.append(f"   {current_line}: {line}")
            code = "\n".join(formatted_lines)

        return {
            "id": node_id,
            "type": data.get('type', 'unknown'),
            "file": node_id.split(':')[0],
            "start_line": start_line,
            "end_line": data.get('end_line', -1),
            "code": code  # May be original code or code with line numbers
        }

    def _locate_specific_node(self, file_path: str, line_num: int):
        """
        Recursive precise location: File -> Class -> Method
        """
        # 1. Normalize path
        if os.path.isabs(file_path):
            rel_file_path = os.path.relpath(file_path, self.repo_path)
        else:
            rel_file_path = file_path

        # If the file node itself does not exist, exit directly
        if not self.entity_searcher.has_node(rel_file_path):
            return None, None

        # 2. Start searching from the file node
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

                        if node['type'] == NODE_TYPE_CLASS:
                            search_queue.append(node['node_id'])
                        elif node['type'] == NODE_TYPE_FUNCTION:
                            search_queue.append(node['node_id'])

        return current_best_node_id, current_best_node_type

    def _find_parent_class(self, node_id):
        parents, _ = self.dep_searcher.get_neighbors(
            node_id, direction='backward', etype_filter=[EDGE_TYPE_CONTAINS]
        )
        for p in parents:
            if self.graph.nodes[p]['type'] == NODE_TYPE_CLASS:
                return p
        return None

    def _is_test_node(self, node_id):
        """Determine if the node belongs to a test file"""
        file_path = node_id.split(':')[0]
        file_path = file_path.replace('\\', '/')
        path_parts = file_path.split('/')
        filename = path_parts[-1]

        if 'tests' in path_parts or 'test' in path_parts:
            return True
        if filename.startswith('test_') or filename.endswith('_test.py'):
            return True
        return False

    def _analyze_line_type(self, line_content: str):
        """Simple analysis of line type"""
        line = line_content.strip()
        if line.startswith("import ") or line.startswith("from "):
            return "IMPORT"
        if line.startswith("@"):
            return "DECORATOR"
        if "=" in line and not line.startswith("def ") and not line.startswith("class "):
            return "VARIABLE_ASSIGNMENT"
        return "OTHER"

    def _extract_call_site(self, source_code: str, target_name: str, node_start_line=0) -> Tuple[str, List[int]]:
        """
        Extracts usage context around a target keyword within source code.

        Args:
            source_code: Node source code
            target_name: Search keyword
            node_start_line: The starting line number of the node in the file (used to calculate absolute line numbers)

        Returns:
            (formatted_full_code, usage_line_nums_list)
        """
        if not source_code:
            return "", []

        lines = source_code.split('\n')
        usage_line_nums = []
        formatted_lines = []

        for i, line in enumerate(lines):
            # Calculate absolute line number
            abs_line = node_start_line + i

            # Check if keyword is matched
            if target_name in line:
                usage_line_nums.append(abs_line)
                prefix = "-->"
            else:
                prefix = "   "

            # Build complete code line with markers and line numbers
            formatted_lines.append(f"{prefix} {abs_line}: {line}")

        # Return formatted complete code
        return "\n".join(formatted_lines), usage_line_nums

    def _traverse_multi_hop(self, start_node: str, direction: str, max_hops: int,
                            edge_filter: List[str]) -> List[Tuple[str, int, str]]:
        """
        Multi-hop search based on BFS
        Returns: List[(node_id, distance, via_node_id)]
        via_node_id: The upstream/downstream node ID that led to the discovery of the current node.
                     Used to determine what keyword to search for in the current node.
        """
        if max_hops < 1:
            return []

        visited = set([start_node])
        # Queue structure: (current_node, level)
        queue = [(start_node, 0)]

        # Results structure: (found_node, distance, via_node)
        results = []

        while queue:
            current_node, level = queue.pop(0)

            if level >= max_hops:
                continue

            search_dir = 'forward' if direction == 'downstream' else 'backward'

            neighbors, _ = self.dep_searcher.get_neighbors(
                current_node,
                direction=search_dir,
                etype_filter=edge_filter,
                ignore_test_file=True
            )

            for neighbor in neighbors:
                if ":" not in neighbor:
                        continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level = level + 1
                    results.append((neighbor, next_level, current_node))

                    queue.append((neighbor, next_level))

        return results
    def get_co_edit_context(self, file_path: str, line_num: int, limit=10, max_hops=2):
        """
        [Core Entry] Get context required for co-edit modifications
        """
        # 1. Locate focus
        focus_id, focus_type = self._locate_specific_node(file_path, line_num)
        if not focus_id:
            return {"error": "Could not locate node"}

        def get_short_name(node_id):
            return node_id.split(':')[-1].split('.')[-1]

        focus_name = focus_id.split(':')[-1].split('.')[-1]
        print(f"📍 Focus Node: [{focus_type}] {focus_id}")

        context_report = {
            "focus_node": self._get_node_details(focus_id, add_line_numbers=True),
            "related_contexts": []
        }

        # --- Internal helper function: uniformly add context ---
        def add_context(node_id, role_base, reason_template, distance, search_keyword=None):
            node_details = self._get_node_details(node_id)
            is_internal = (node_details['file'] == file_path)

            # Construct Role: if indirect dependency, prepend INDIRECT prefix
            prefix = "INTERNAL" if is_internal else "EXTERNAL"
            dist_prefix = "INDIRECT_" if distance > 1 else ""
            full_role = f"{prefix}_{dist_prefix}{role_base}".upper()

            # Construct Reason: distinguish direct vs indirect
            dist_desc = "Directly" if distance == 1 else f"Indirectly (hop {distance})"
            reason = reason_template.format(
                name=search_keyword,
                # loc="same file" if is_internal else "other file",
                how=dist_desc
            )

            if search_keyword:
                snippet, usage_line_nums = self._extract_call_site(node_details['code'], search_keyword,
                                                     node_start_line=node_details['start_line'])
            else:
                # snippet truncated to first 10 lines if needed
                snippet = "\n".join(node_details['code'].split('\n'))
                usage_line_nums = []
            prefix_len = sum(1 for _ in takewhile(lambda x: x[0] == x[1], zip(file_path, node_details['file'])))
            context_report["related_contexts"].append({
                "role": full_role,
                "reason": reason,
                "distance": distance,
                "node_id": node_id,
                "file_path": node_details['file'],
                "rel_file_path" : node_details['file'][prefix_len:],
                "relevant_code": snippet,
                'usage_line_nums': usage_line_nums
            })

        # --- Helper function: filter and truncate ---
        def slice_results(results_list):
            valid = [n for n in results_list if not self._is_test_node(n[0])]
            # 2. Then take at most limit items
            return valid[:limit]
        # =========================================================
        # Strategy A: Modified entity is a function
        # =========================================================
        if focus_type == NODE_TYPE_FUNCTION:
            # 1. Callers (Upstream)
            callers = self._traverse_multi_hop(
                focus_id, direction='upstream', max_hops=max_hops
                ,edge_filter = []
                # , edge_filter=[EDGE_TYPE_INVOKES]
            )

            # [Key] Unpack three values: node_id, dist, via_node
            for node_id, dist, via_node in slice_results(callers):
                # Dynamic keyword: if Hop 1, via_node is focus_id, search for focus_name
                # If Hop 2, via_node is the intermediate node, search for the intermediate node's name
                search_key = get_short_name(via_node)

                add_context(node_id, "CALLER", "{how} calls `{name}`.", dist, search_keyword=search_key)

            # 2. Callees (Downstream)
            callees = self._traverse_multi_hop(
                focus_id, direction='downstream', max_hops=max_hops
                ,edge_filter = []
                # , edge_filter=[EDGE_TYPE_INVOKES]
            )
            for node_id, dist, via_node in slice_results(callees):
                # Downstream usually does not need keyword search (showing the callee's definition is sufficient),
                # or searching who calls it (less common). Keep None here, or adjust as needed.
                add_context(node_id, "CALLEE", "{how} called by `{name}`.", dist, search_keyword=None)

        # =========================================================
        # Strategy B: Modified entity is a class
        # =========================================================
        elif focus_type == NODE_TYPE_CLASS:
            # 1. Subclasses
            subclasses = self._traverse_multi_hop(
                focus_id, direction='upstream', max_hops=max_hops, edge_filter=[EDGE_TYPE_INHERITS]
            )
            for node_id, dist, via_node in slice_results(subclasses):
                search_key = get_short_name(via_node)
                add_context(node_id, "SUBCLASS", "{how} inherits from `{name}`.", dist,
                            search_keyword=search_key)

            # 2. Users
            users = self._traverse_multi_hop(
                focus_id, direction='upstream', max_hops=max_hops, edge_filter=[EDGE_TYPE_INVOKES]
            )
            for node_id, dist, via_node in slice_results(users):
                search_key = get_short_name(via_node)
                add_context(node_id, "USER", "{how} uses `{name}`.", dist, search_keyword=search_key)

            # 3. Methods (unchanged, since they are direct children)
            methods, _ = self.dep_searcher.get_neighbors(focus_id, direction='forward',
                                                         etype_filter=[EDGE_TYPE_CONTAINS])
            valid_methods = [m for m in methods if not self._is_test_node(m)][:limit]
            for method_id in valid_methods:
                add_context(method_id, "METHOD", "Method of `{name}`.", distance=1, search_keyword=None)

        # =========================================================
        # Strategy C: Modified entity is at file level (variable/constant)
        # =========================================================
        elif focus_type == NODE_TYPE_FILE:
            file_details = self._get_node_details(focus_id)
            lines = file_details['code'].split('\n')
            target_line = lines[line_num - 1] if 0 < line_num <= len(lines) else ""

            var_name = None
            if "=" in target_line and not target_line.strip().startswith(("def ", "class ", "import ", "from ")):
                var_name = target_line.split('=')[0].strip().split(':')[0].strip()

            if var_name:
                print(f"📂 Variable Change Detected: {var_name}")

                # 1. [Internal] Scan functions in the same file (single level, since they are within the file)
                children, _ = self.dep_searcher.get_neighbors(focus_id, direction='forward',
                                                              etype_filter=[EDGE_TYPE_CONTAINS])
                for child_id in children:
                    child_details = self._get_node_details(child_id)
                    prefix_len = sum(1 for _ in takewhile(lambda x: x[0] == x[1], zip(file_path, child_details['file'])))
                    if var_name in child_details['code']:
                        snippet, usage_line_nums = self._extract_call_site(child_details['code'], var_name, node_start_line=child_details['start_line'])
                        context_report["related_contexts"].append({
                            "role": "INTERNAL_USAGE",
                            "reason": f"Uses global variable `{var_name}`.",
                            "distance": 1,
                            "node_id": child_id,
                            "file_path": file_path,
                            "rel_file_path" : file_path[prefix_len:],
                            "relevant_code": snippet,
                            'usage_line_nums': usage_line_nums,
                        })

                # 2. [External] Importers (supports multi-hop: A import B, C import A)
                importers = self._traverse_multi_hop(
                    focus_id, direction='upstream', max_hops=max_hops, edge_filter=[EDGE_TYPE_IMPORTS]
                )
                for node_id, dist, via_node in slice_results(importers):
                    # For imports, typically search for the imported module name or variable name
                    # If Hop 1, via_node is the file, search for var_name
                    # If Hop 2, via_node is the intermediate file, search for the intermediate module name (simplified below)

                    # Simple strategy: if direct import, search variable name; if indirect, search intermediate module name
                    if dist == 1:
                        key = var_name
                    else:
                        # Try to search the filename of the intermediate module (excluding extension)
                        key = get_short_name(via_node)

                    add_context(node_id, "IMPORTER", "{how} imports modified file.", dist, search_keyword=key)
        return context_report

    def check_context_coverage(
            self,
            analysis_result: dict,
            ground_truths,  # List[GroundTruth]
    ):
        """
        Verify whether Ground Truth is covered by context_report.

        Matching strategy (AND):
          1. File-level match: GT.rel_file_path is in the collected file set
          2. Code-level match: GT.before_code (after strip) is a substring of
                               some context's relevant_code

        Returns:
            ContextCoverageResult
        """
        from src.domain.types import ContextCoverageResult, GTCoverageDetail

        # ── 1. Build collected context index ──────────────────────────────
        # { rel_file_path → List[context_dict] }
        collected_index = {}

        # Also include the focus_node itself (corresponds to idx=0 case)
        focus_node = analysis_result.get("focus_node", {})
        focus_file = focus_node.get("file", "")
        if focus_file:
            # focus_node's code is the complete node code with line number markers
            collected_index.setdefault(focus_file, []).append({
                "role": "Focus Node",
                "node_id": focus_node.get("id", ""),
                "relevant_code": focus_node.get("code", ""),
            })

        for ctx in analysis_result.get("related_contexts", []):
            # Compatibility: handle inconsistent file_path / file fields in FILE strategy
            fp = ctx.get("file_path") or ctx.get("file", "")
            if fp:
                collected_index.setdefault(fp, []).append({
                    "role": ctx.get("role", ""),
                    "node_id": ctx.get("node_id", ""),
                    "relevant_code": ctx.get("relevant_code", ""),
                })

        logger.debug(f"📊 Collected files: {list(collected_index.keys())}")

        # ── 2. Verify each GT entry ────────────────────────────────────────
        details = []

        for gt in ground_truths:
            gt_file = gt.rel_file_path
            # Strip before_code to avoid false negatives due to leading/trailing whitespace
            gt_code = gt.before_code.strip() if gt.before_code else ""

            detail = GTCoverageDetail(
                gt_rel_file_path=gt_file,
                gt_start_line=gt.start_line,
                gt_end_line=gt.end_line,
                file_matched=False,
                code_matched=False,
                is_covered=False,
            )

            # ── 2a. File-level match ──────────────────────────────────
            # Normalize path: use / as separator, ignore leading/trailing slashes
            def norm_path(p: str) -> str:
                return p.replace("\\", "/").strip("/")

            gt_norm = norm_path(gt_file)
            matched_contexts = []

            for collected_file, ctx_list in collected_index.items():
                if norm_path(collected_file) == gt_norm:
                    detail.file_matched = True
                    matched_contexts = ctx_list
                    break

            if not detail.file_matched:
                detail.miss_reason = (
                    f"File '{gt_file}' not found in collected contexts. "
                    f"Collected: {[norm_path(f) for f in collected_index.keys()]}"
                )
                details.append(detail)
                logger.warning(
                    f"❌ [Coverage] File miss: '{gt_file}' not in collected contexts.\n"
                    f"   Collected files: {list(collected_index.keys())}"
                )
                continue

            # ── 2b. Code-level match ──────────────────────────────────
            if not gt_code:
                # If before_code is empty, file match alone counts as coverage
                detail.code_matched = True
                detail.is_covered = True
                detail.miss_reason = ""
                if matched_contexts:
                    detail.matched_context_role = matched_contexts[0]["role"]
                    detail.matched_node_id = matched_contexts[0]["node_id"]
                details.append(detail)
                continue

            code_hit = False
            for ctx in matched_contexts:
                relevant = ctx.get("relevant_code", "")
                # relevant_code has line number markers ("-->  42: code"),
                # strip the prefix from each line before matching
                stripped_relevant = self._strip_line_markers(relevant)

                if gt_code in stripped_relevant:
                    code_hit = True
                    detail.code_matched = True
                    detail.is_covered = True
                    detail.matched_context_role = ctx["role"]
                    detail.matched_node_id = ctx["node_id"]
                    detail.miss_reason = ""
                    break

            if not code_hit:
                detail.miss_reason = (
                    f"File matched ('{gt_file}'), but before_code not found "
                    f"in any relevant_code snippet. "
                    f"GT code (first 80 chars): '{gt_code[:80]}...'"
                )
                logger.warning(
                    f"⚠️ [Coverage] Code miss in '{gt_file}': "
                    f"before_code not found in relevant_code snippets.\n"
                    f"   GT code preview: {repr(gt_code[:100])}"
                )

            details.append(detail)

        # ── 3. Aggregate results ───────────────────────────────────────────
        covered_count = sum(1 for d in details if d.is_covered)
        total_gt = len(details)
        recall = covered_count / total_gt if total_gt > 0 else 1.0
        is_framework_issue = covered_count < total_gt

        result = ContextCoverageResult(
            total_gt=total_gt,
            covered_count=covered_count,
            recall=recall,
            is_framework_issue=is_framework_issue,
            details=details,
        )

        # Print summary
        status = "✅" if not is_framework_issue else "🚨"
        logger.info(
            f"{status} [Coverage] Recall={recall:.2%} "
            f"({covered_count}/{total_gt} GT covered)"
        )
        if is_framework_issue:
            logger.warning(
                f"🚨 [Framework Issue] {total_gt - covered_count} GT(s) NOT covered. "
                f"LLM call will be SKIPPED.\n"
                f"   Summary: {result.summary()}"
            )

        return result

    def _strip_line_markers(self,relevant_code: str) -> str:
        """
        Strip line number markers from code and return clean code string for matching.

        Input formats (two types):
          "-->  42: def foo():"      ← _extract_call_site output
          "   42: def foo():"        ← _get_node_details add_line_numbers=True output

        Output:
          "def foo():"
        """
        lines = relevant_code.split("\n")
        clean_lines = []
        for line in lines:
            # Match "-->  42: " or "    42: " prefix
            # Format: (-->|   ) + spaces* + digits+ + ': ' + actual code
            import re
            m = re.match(r'^(?:-->|   )\s*\d+:\s?(.*)', line)
            if m:
                clean_lines.append(m.group(1))
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines)

