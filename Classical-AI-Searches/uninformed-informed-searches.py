from typing import Optional, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque #Double Ended Queue
import heapq
import unittest
import numpy as np
from io import StringIO
import sys


def _validate_inputs(G, start, end): 
    """Common input validation for all search algorithms."""
    if start not in G or end not in G:
        print("Start or goal not in graph. Choose valid nodes.")
        return False
    return True

def _initialize_search_state():
    """Initialize common search state variables."""
    return {
        'visited': set(), #to record all the visited nodes
        'expanded': [], #keeps a track of all expanded nodes
        'tree_edges': [], 
        'max_depth': 0, #shows the depth of the graph we visited so far
        'step': 0
    }

def _print_frontier_state(step, frontier_data, algorithm):
    """Print current frontier state for any algorithm."""
    print(f"\n--- Step {step} ---")
    if algorithm == 'BFS' or algorithm == 'DFS':
        nodes, depths = frontier_data
        print(f"Frontier queue/stack: {nodes} (depths: {depths}) | size={len(nodes)}")        
    elif algorithm == 'UCS':
        frontier_info = frontier_data
        # frontier_info is a list of (cost, node, path, depth, parent)
        # We'll just show the (cost, node) for brevity in the printout
        display_info = [(f"{c:.1f}", n) for (c, n, _, _, _) in frontier_info]
        print(f"Priority queue (cost-ordered): {display_info} | size={len(frontier_info)}")

def _print_goal_reached(path, max_depth, max_size, cost=None):
    """Print goal reached message with statistics."""
    print(f"\nGOAL REACHED!")
    print(f"Path: {' → '.join(path)}")
    if cost is not None:
        print(f"Total cost: ${cost:.1f}")
    print(f"Path length: {len(path)-1} edges")
    print(f"Max depth explored: {max_depth}")
    print(f"Peak memory usage: {max_size} nodes in {'queue' if cost is not None else 'queue/stack'}")

def _print_goal_not_found(end, start, max_depth, max_size):
    """Print goal not found message with statistics."""
    print(f"\n❌ Goal '{end}' not reachable from '{start}'")
    print(f"Max depth explored: {max_depth}")
    print(f"Peak memory usage: {max_size} nodes in queue")


def breadth_first_search(
    G: nx.Graph, 
    start: str, 
    end: str
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Perform Breadth-First Search (BFS) to find a path from start to end node.
    
    BFS explores nodes level by level using a queue (FIFO structure).
    
    Args:
        G: A NetworkX graph
        start: The starting node label
        end: The goal node label
    
    Returns:
        tuple: (path, expanded_nodes, tree_edges, max_depth)
            - path: List of nodes from start to end, or None if no path exists
            - expanded_nodes: List of nodes explored during search
            - tree_edges: List of (parent, child) tuples forming the search tree
            - max_depth: Maximum depth reached during search
    """
    if not _validate_inputs(G, start, end):
        return None, [], [], 0 #says no path, no expanded nodes, no tree_edges
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement BFS algorithm
    # Hint: You'll need a queue and a way to track visited nodes
    # Hint: Think about what information you need to store for each node

    # Initializing the queue
    # Queue stores: (node, path_to_node, depth, parent_node)
    queue = deque([(start, [start], 0, None)])
    state = _initialize_search_state()
    max_queue_size = 1
    # 'visited' set will track nodes that have been added to the queue to avoid cycles/redundancy

    print(f"Starting BFS from {start} to {end}")

    # Iterating over the queue to the get the current node 
    while queue:
        state['step'] += 1
        max_queue_size = max(max_queue_size, len(queue))

        # show the frontier state and expand the node
        queue_nodes = [n for (n,_,_,_) in queue]
        depths = [d for (_,_,d,_) in queue]
        _print_frontier_state(state['step'], (queue_nodes, depths), 'BFS')
                        
        # pop the first element of the queue (FIFO for BFS)
        node, path, depth, parent = queue.popleft()
        print("Popped node:", node)

        # check if the element has already been expanded/visited (optional for trees, crucial for graphs)
        if node in state["visited"]:
            # Note: For BFS, we usually put a node in 'visited' only when we EXPAND it.
            # If we check 'visited' upon insertion, it guarantees the shortest path (in hops)
            # is found first. Since we are using an 'expanded' set, let's keep the check here
            # to be consistent with many textbook implementations.
            print(f"Skipping {node} as it's already visited/expanded")
            continue
        
        # Now we expand the node
        print(f"Expanding {node} at depth {depth}")

        state["visited"].add(node)
        state["expanded"].append(node)
        state["max_depth"] = max(state["max_depth"], depth)
        if parent is not None:
            state['tree_edges'].append((parent, node))

        if node == end:
            _print_goal_reached(path, state['max_depth'], max_queue_size) 
            return path, state['expanded'], state['tree_edges'], state['max_depth']
            
        # get the neighbors of the current node
        neighbors = list(G.neighbors(node))
        
        # Filter for unvisited neighbors. Crucial to prevent cycles and re-exploring.
        unvisited_neighbors = [n for n in neighbors if n not in state['visited']]
        
        # Adding to the queue *must* be done carefully: a better approach for BFS
        # in a general graph is to check 'visited' *before* adding to the queue.
        # But for this template, let's stick to the check on expansion,
        # and just ensure we don't re-add a node that is already expanded.
        # We can adjust the logic for consistency if needed, but the current
        # structure works if 'visited' means 'expanded'.

        # Print state update
        print(f"Neighbors: {neighbors}")
        print(f"Unvisited Neighbors: {unvisited_neighbors}")
        print(f"Enqueuing successors at depth {depth+1}")

        # add neighbors to the queue
        for neighbor in neighbors: # Using all neighbors is safe if 'visited' check is upon expansion
            if neighbor not in state['visited']:
                # The path to the neighbor is the current path + the neighbor
                new_path = path + [neighbor]
                new_depth = depth + 1
                queue.append((neighbor, new_path, new_depth, node))

    
    # ===== END YOUR CODE =====
    
    _print_goal_not_found(end, start, state["max_depth"], max_queue_size)
    return None, state["expanded"], state["tree_edges"], state["max_depth"]


def depth_first_search(
    G: nx.Graph, 
    start: str, 
    end: str
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Perform Depth-First Search (DFS) to find a path from start to end node.
    
    DFS explores as deep as possible along each branch before backtracking,
    using a stack (LIFO structure).
    
    Args:
        G: A NetworkX graph
        start: The starting node label
        end: The goal node label
    
    Returns:
        tuple: (path, expanded_nodes, tree_edges, max_depth)
            - path: List of nodes from start to end, or None if no path exists
            - expanded_nodes: List of nodes explored during search
            - tree_edges: List of (parent, child) tuples forming the search tree
            - max_depth: Maximum depth reached during search
    """
    if not _validate_inputs(G, start, end):
        return None, [], [], 0
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement DFS algorithm (iterative, not recursive)
    # Hint: Use a stack (LIFO) instead of a queue
    # Hint: Think about how this differs from BFS

    # Stack stores: (node, path_to_node, depth, parent_node)
    # Using a list as a stack (append for push, pop for pop)
    stack = [(start, [start], 0, None)]
    state = _initialize_search_state()
    max_stack_size = 1

    print(f"Starting DFS from {start} to {end}")

    while stack:
        state['step'] += 1
        max_stack_size = max(max_stack_size, len(stack))

        # show the frontier state and pop the node
        # Note: DFS typically shows nodes in the order they will be explored (LIFO)
        stack_nodes = [n for (n,_,_,_) in stack]
        depths = [d for (_,_,d,_) in stack]
        _print_frontier_state(state['step'], (stack_nodes, depths), 'DFS')

        # Pop the last element (LIFO for DFS)
        node, path, depth, parent = stack.pop()
        print("Popped node:", node)

        # Check for cycles/re-expansion
        if node in state["visited"]:
            print(f"Skipping {node} as it's already visited/expanded")
            continue

        # Now we expand the node
        print(f"Expanding {node} at depth {depth}")

        state["visited"].add(node)
        state["expanded"].append(node)
        state["max_depth"] = max(state["max_depth"], depth)
        if parent is not None:
            state['tree_edges'].append((parent, node))

        if node == end:
            _print_goal_reached(path, state['max_depth'], max_stack_size) 
            return path, state['expanded'], state['tree_edges'], state['max_depth']
            
        # Get the neighbors. DFS often adds neighbors in reverse order
        # (or some fixed order) so the desired neighbor is popped first.
        # Since NetworkX's neighbors() order is not guaranteed, and
        # to ensure correct path finding, we will iterate and push.
        neighbors = list(G.neighbors(node))
        
        # Reversing neighbors list before pushing to stack ensures we explore
        # them in the standard iteration order (if the graph structure permits)
        # to match typical DFS behavior/examples (e.g., A's neighbors are B, C.
        # We want to visit B first, so we push C then B to the stack).
        neighbors.reverse() 
        
        unvisited_neighbors = [n for n in neighbors if n not in state['visited']]

        print(f"Neighbors: {neighbors}")
        print(f"Unvisited Neighbors (will be pushed): {unvisited_neighbors}")
        print(f"Pushing successors at depth {depth+1}")

        # Add neighbors to the stack
        for neighbor in neighbors:
             # Check before pushing, which is a common and safer way for DFS on a general graph
            if neighbor not in state['visited']:
                new_path = path + [neighbor]
                new_depth = depth + 1
                stack.append((neighbor, new_path, new_depth, node))
        
    # ===== END YOUR CODE =====
    
    _print_goal_not_found(end, start, state["max_depth"], max_stack_size)
    return None, state["expanded"], state["tree_edges"], state["max_depth"]


def uniform_cost_search(
    G: nx.Graph, 
    start: str, 
    end: str,
    distance_matrix=None
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int, Optional[float]]:
    """
    Perform Uniform Cost Search (UCS) to find the lowest-cost path from start to end.
    
    UCS expands nodes in order of path cost, guaranteeing an optimal solution.
    Uses a priority queue where priority is the cumulative path cost.
    
    Args:
        G: A NetworkX graph with 'weight' attributes on edges
        start: The starting node label
        end: The goal node label
        distance_matrix: Optional numpy array of edge costs (if provided, used instead of edge weights)
    
    Returns:
        tuple: (path, expanded_nodes, tree_edges, max_depth, total_cost)
            - path: List of nodes from start to end, or None if no path exists
            - expanded_nodes: List of nodes explored during search
            - tree_edges: List of (parent, child) tuples forming the search tree
            - max_depth: Maximum depth reached during search
            - total_cost: Cost of the path found, or None if no path exists
    """
    if not _validate_inputs(G, start, end):
        return None, [], [], 0, 0
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement UCS algorithm
    # Hint: Use heapq for the priority queue (heappush, heappop)
    # Hint: Priority should be cumulative path cost
    # Hint: Edge costs come from either distance_matrix or G[node][neighbor]["weight"]
    
    # Priority Queue stores: (cost, node, path_to_node, depth, parent_node)
    # The cost is the primary key for min-heap.
    pq = [(0.0, start, [start], 0, None)]
    state = _initialize_search_state()
    max_pq_size = 1
    
    # Track the lowest cost found so far to reach a node (Dijkstra-like)
    cost_so_far = {start: 0.0}
    
    # Map nodes to indices if using a distance matrix
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    
    def get_edge_cost(u, v):
        """Helper to get edge cost from graph or distance matrix."""
        if distance_matrix is not None:
            u_idx = node_to_index.get(u)
            v_idx = node_to_index.get(v)
            if u_idx is not None and v_idx is not None:
                return distance_matrix[u_idx, v_idx]
            else:
                raise ValueError(f"Node {u} or {v} not found in distance matrix map.")
        else:
            # Fallback to NetworkX edge weight, default to 1 if no 'weight'
            return G[u][v].get('weight', 1.0)

    print(f"Starting UCS from {start} to {end}")

    while pq:
        state['step'] += 1
        max_pq_size = max(max_pq_size, len(pq))

        # Show the frontier state
        _print_frontier_state(state['step'], pq, 'UCS')
        
        # Pop the element with the lowest cost (priority queue)
        cost, node, path, depth, parent = heapq.heappop(pq)
        print(f"Popped node: {node} with cumulative cost: ${cost:.1f}")

        # UCS check for already expanded nodes:
        # If we found a cheaper path to a node already expanded, we ignore the current one.
        # But since we use Dijkstra's structure (only storing the best cost to a node),
        # we check the cost_so_far dictionary.
        # This check also serves as the 'visited' check for expansion.
        if node in state["expanded"]:
            if cost > cost_so_far.get(node, float('inf')):
                 # We found a more expensive way to an already expanded node, so skip.
                print(f"Skipping {node} as a cheaper path (cost ${cost_so_far[node]:.1f}) was already expanded.")
                continue
            # If cost == cost_so_far, it's a tie-break path that was just as good.
            # We can choose to skip or process. For simplicity, we process only the
            # first time a node is *expanded* with its optimal cost.
            
        # Expand the node
        print(f"Expanding {node} at depth {depth}")

        # Update state:
        state["expanded"].append(node)
        state["max_depth"] = max(state["max_depth"], depth)
        if parent is not None:
            state['tree_edges'].append((parent, node))

        # Goal check (must be on expansion for optimal path guarantee in UCS)
        if node == end:
            _print_goal_reached(path, state['max_depth'], max_pq_size, cost) 
            return path, state['expanded'], state['tree_edges'], state['max_depth'], cost

        # Explore neighbors
        neighbors = list(G.neighbors(node))
        print(f"Neighbors: {neighbors}")
        print(f"Processing successors at depth {depth+1}")

        for neighbor in neighbors:
            edge_cost = get_edge_cost(node, neighbor)
            new_cost = cost + edge_cost
            
            # Use a Dijkstra-like condition: if a cheaper path is found, update/add to PQ
            if new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                new_path = path + [neighbor]
                new_depth = depth + 1
                # Push to priority queue: (new_cost, neighbor, new_path, new_depth, node)
                heapq.heappush(pq, (new_cost, neighbor, new_path, new_depth, node))
                print(f"  Pushed {neighbor} (Cost: ${new_cost:.1f}, Parent: {node})")
            else:
                print(f"  Skipped {neighbor} (Current Cost: ${new_cost:.1f} vs Known Best: ${cost_so_far.get(neighbor, float('inf')):.1f})")


    # ===== END YOUR CODE =====
    
    _print_goal_not_found(end, start, state["max_depth"], max_pq_size)
    return None, state["expanded"], state["tree_edges"], state["max_depth"], None


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Recursive hierarchy pos for tree drawing."""
    pos = {root:(xcenter,vert_loc)}
    children = list(G.neighbors(root))
    if children:
        dx = width/len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos.update(hierarchy_pos(G,child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

def visualize_search_tree(tree_edges, start, end, path, expanded, 
                          title="Search Exploration Tree", 
                          show_distances=False, distance_matrix=None, nodes=None):
    """Visualize a search exploration tree (DFS, BFS, or UCS). Optionally show edge distances/costs."""
    T = nx.DiGraph()
    T.add_edges_from(tree_edges)

    if not T.nodes():
        print("No tree to visualize.")
        return

    pos = hierarchy_pos(T, start)
    colors = {"start": "green", "goal": "red", "path": "orange", "expanded": "yellow"}
    
    # Determine node colors
    node_colors = [colors["start"] if n == start else colors["goal"] if n == end 
                   else colors["path"] if path and n in path else colors["expanded"] 
                   if n in expanded else "lightblue" for n in T.nodes()]

    plt.figure(figsize=(10, 8))
    
    nx.draw_networkx_edges(T, pos, arrows=True, arrowstyle="-|>", arrowsize=18,
                          min_target_margin=12, connectionstyle="arc3,rad=0.06", 
                          width=1.5, edge_color="dimgray")
    nx.draw_networkx_nodes(T, pos, node_color=node_colors, node_size=700, 
                          edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(T, pos, font_weight="bold")

    # Show distances if requested
    if show_distances:
        if distance_matrix is not None and nodes is not None:
            node_index = {node: i for i, node in enumerate(nodes)}
            labels = {(u, v): round(distance_matrix[node_index[u], node_index[v]], 2) 
                     for u, v in T.edges() if u in node_index and v in node_index}
        else:
            labels = nx.get_edge_attributes(T, "weight")
        nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, font_color="blue")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                             markerfacecolor=color, markeredgecolor='black', markersize=12)
                      for label, color in [('Start', colors["start"]), ('Goal', colors["goal"]),
                                          ('On solution path', colors["path"]), ('Expanded', colors["expanded"])]]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, title="Legend")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# UNIT TESTS FOR SEARCH ALGORITHMS
# =============================================================================

class TestSearchAlgorithms(unittest.TestCase):
    """Starter tests: one placeholder per algorithm. Students must implement."""

    def setUp(self):
        """Set up any graphs needed for tests.
        Hints:
        - Create a small unweighted graph for BFS/DFS.
        - Create a small weighted graph for UCS (use 'weight' attributes).
        - Consider adding a disconnected graph for later tests you write.
        """
        # TODO: Initialize graphs like: (just as examples)
        
        # Unweighted graph for BFS/DFS
        self.unweighted_graph = nx.Graph()
        self.unweighted_graph.add_edges_from([
            ('S', 'A'), ('S', 'B'),
            ('A', 'C'), ('A', 'D'),
            ('B', 'D'), ('B', 'E'),
            ('C', 'G'), ('D', 'G'),
            ('E', 'F'), ('F', 'G')
        ])
        
        # Weighted graph for UCS
        self.weighted_graph = nx.Graph()
        # Path 1: S->A->G (cost 1+10 = 11)
        # Path 2: S->B->G (cost 5+3 = 8) - OPTIMAL
        # Path 3: S->B->F->G (cost 5+1+1 = 7) - Shorter path in hops but higher cost
        self.weighted_graph.add_edge('S', 'A', weight=1)
        self.weighted_graph.add_edge('S', 'B', weight=5)
        self.weighted_graph.add_edge('A', 'G', weight=10)
        self.weighted_graph.add_edge('B', 'G', weight=3) 
        self.weighted_graph.add_edge('B', 'C', weight=1)
        self.weighted_graph.add_edge('C', 'G', weight=1)
        # Re-defining to make the optimal path less obvious: S->B->C->G (5+1+1=7)
        # Let's use the one from the comment hint that shows cost != hops:
        # S->A->G: cost 11, hops 2
        # S->B->G: cost 8, hops 2
        # S->B->C->G: cost 5+1+1 = 7, hops 3
        # Optimal cost is 7.
        
        self.weighted_graph = nx.Graph()
        self.weighted_graph.add_edge('S', 'A', weight=10) # 10
        self.weighted_graph.add_edge('S', 'B', weight=1)  # 1
        self.weighted_graph.add_edge('A', 'G', weight=1)  # 11
        self.weighted_graph.add_edge('B', 'C', weight=1)  # 2
        self.weighted_graph.add_edge('C', 'G', weight=1)  # 3 <- Optimal
        
        # Disconnected graph for unreachable tests
        self.disconnected_graph = nx.Graph()
        self.disconnected_graph.add_edges_from([('A', 'B'), ('C', 'D')])

    def tearDown(self):
        """Optional cleanup."""
        pass

    def suppress_output(self):
        """Use this to silence prints while testing."""
        return SuppressOutput()

    # ==================== BFS Starter ====================

    def test_bfs_reaches_goal(self):
        """BFS: ensure a path from start to goal is found on an unweighted graph.
        Hints:
        - Call breadth_first_search on a small graph where a path exists.
        - Assert that a path is returned and it starts/ends at the expected nodes.
        - Optionally, later add an assertion that the path length equals the shortest distance.
        """
        # TODO: Implement the test body
        with self.suppress_output():
            path, expanded, _, max_depth = breadth_first_search(self.unweighted_graph, 'S', 'G')
        
        # Assert a path was found
        self.assertIsNotNone(path, "BFS failed to find a path from S to G.")
        
        # Assert the path is correct (shortest path length is 2: S-C-G or S-D-G)
        self.assertEqual(path[0], 'S')
        self.assertEqual(path[-1], 'G')
        self.assertEqual(len(path) - 1, 2, "BFS did not find the shortest path in terms of hops.")
        self.assertIn(path[1], ['A', 'B'], "BFS path is not valid.")
        
        # Assert maximum depth is correct
        self.assertEqual(max_depth, 2, "Max depth explored should be 2 for a shortest path length of 2.")
        
    def test_bfs_unreachable_goal(self):
        """BFS: Test case where the goal is unreachable."""
        with self.suppress_output():
            path, _, _, _ = breadth_first_search(self.disconnected_graph, 'A', 'C')
        self.assertIsNone(path, "BFS found a path between A and C in a disconnected graph.")

    # ==================== DFS Starter ====================

    def test_dfs_finds_a_path(self):
        """DFS: ensure some valid path is found (not necessarily shortest).
        Hints:
        - Call depth_first_search on the same unweighted graph.
        - Assert that a path exists and each consecutive pair forms an edge in the graph.
        - Later, add tests demonstrating deeper-first exploration and unreachable cases.
        """
        # TODO: Implement the test body
        with self.suppress_output():
            path, expanded, _, max_depth = depth_first_search(self.unweighted_graph, 'S', 'G')

        # Assert a path was found
        self.assertIsNotNone(path, "DFS failed to find a path from S to G.")
        
        # Assert the path is valid (starts at S, ends at G)
        self.assertEqual(path[0], 'S')
        self.assertEqual(path[-1], 'G')
        
        # Assert that all steps in the path are valid edges
        for i in range(len(path) - 1):
            self.assertTrue(self.unweighted_graph.has_edge(path[i], path[i+1]), 
                            f"DFS path step {path[i]}->{path[i+1]} is not a valid edge.")

    def test_dfs_deeper_exploration(self):
        """DFS: Ensure DFS explores deeper paths before finding the goal."""
        # Using a graph where a long path is explored first
        G = nx.Graph()
        G.add_edges_from([('S', 'A'), ('S', 'B'), ('A', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'G'), ('B', 'G')])
        
        with self.suppress_output():
            path, expanded, _, max_depth = depth_first_search(G, 'S', 'G')
            
        # If neighbors are explored in alphabetical order, DFS will go S-A-C-D-E-G
        expected_path = ['S', 'A', 'C', 'D', 'E', 'G']
        self.assertEqual(path, expected_path, "DFS did not follow the expected deeper path first.")
        self.assertEqual(len(path) - 1, 5, "Path length should be 5 for S-A-C-D-E-G.")
        self.assertEqual(max_depth, 5, "Max depth should be 5.")
        
    # ==================== UCS Starter ====================

    def test_ucs_optimal_cost(self):
        """UCS: ensure the returned path cost is optimal on a weighted graph.
        Hints:
        - Build a small weighted graph where the cheapest route is not the shortest by hops.
        - Call uniform_cost_search and assert the total cost equals the known optimum.
        - Later, add tests that use a distance_matrix, invalid inputs, and tie-breaking.
        """
        # TODO: Implement the test body
        
        # The weighted graph is designed to have optimal cost path: S->B->C->G (Cost 1+1+1 = 3)
        # and a shorter path in hops: S->A->G (Cost 10+1 = 11)
        
        with self.suppress_output():
            path, expanded, _, max_depth, cost = uniform_cost_search(self.weighted_graph, 'S', 'G')

        # Assert a path was found
        self.assertIsNotNone(path, "UCS failed to find a path from S to G.")
        self.assertIsNotNone(cost, "UCS failed to return a path cost.")
        
        # Assert the optimal cost is found
        optimal_cost = 3.0
        self.assertAlmostEqual(cost, optimal_cost, delta=1e-6, msg=f"UCS failed to find optimal cost. Found {cost}, expected {optimal_cost}")
        
        # Assert the optimal path is correct
        optimal_path = ['S', 'B', 'C', 'G']
        self.assertEqual(path, optimal_path, f"UCS found path {path}, expected optimal path {optimal_path}.")
        
        # Assert maximum depth is correct (3 edges)
        self.assertEqual(max_depth, 3, "Max depth explored should be 3.")
        
    def test_ucs_distance_matrix(self):
        """UCS: Test using a distance matrix instead of edge weights."""
        
        # Create a tiny graph for the matrix test
        nodes = ['A', 'B', 'C']
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        # Matrix: A->B cost 2, A->C cost 10, B->C cost 1
        # Optimal A->C is A->B->C (2+1 = 3)
        dist_matrix = np.array([
            [0, 2, 10],  # A
            [2, 0, 1],   # B
            [10, 1, 0]   # C
        ])
        
        with self.suppress_output():
            path, _, _, _, cost = uniform_cost_search(G, 'A', 'C', distance_matrix=dist_matrix)
            
        optimal_cost = 3.0
        optimal_path = ['A', 'B', 'C']
        
        self.assertIsNotNone(path)
        self.assertAlmostEqual(cost, optimal_cost)
        self.assertEqual(path, optimal_path)


class SuppressOutput:
    """Context manager to suppress stdout."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def run_search_tests():
    """Run all search algorithm unit tests."""
    print("=" * 60)
    print("RUNNING SEARCH ALGORITHM UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSearchAlgorithms)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL SEARCH ALGORITHM TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME SEARCH ALGORITHM TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        # Print failure details
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                # Truncate traceback to show only the assertion message
                detail = traceback.split('AssertionError:')[-1].strip().split('\n')[0]
                print(f"- {test}: {detail}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                # Truncate traceback to show only the exception
                detail = traceback.split('Exception:')[-1].strip().split('\n')[0]
                print(f"- {test}: {detail}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Only run tests when script is executed directly
    success = run_search_tests()
    sys.exit(0 if success else 1)
