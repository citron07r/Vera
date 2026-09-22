//! Multi-hop graph traversal and path analysis over code symbol dependencies.
//!
//! Provides bounded, cycle-safe BFS/DFS graph traversals to extract:
//! 1. Multi-hop caller and callee dependency chains.
//! 2. Shortest dependency paths between symbols.
//! 3. Subgraph neighborhood clusters for context expansion.

use std::collections::{HashMap, HashSet, VecDeque};

/// Direction of graph edge traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Outgoing edges: symbol calls or uses dependencies (callees).
    Outgoing,
    /// Incoming edges: symbols that call or depend on this symbol (callers).
    Incoming,
}

/// A directed edge between code entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolEdge {
    pub from: String,
    pub to: String,
    pub kind: String,
}

/// In-memory symbol dependency graph for multi-hop structural traversals.
#[derive(Debug, Clone, Default)]
pub struct SymbolGraph {
    /// Adjacency list: from -> list of (to, edge_kind)
    outgoing: HashMap<String, Vec<(String, String)>>,
    /// Reverse adjacency list: to -> list of (from, edge_kind)
    incoming: HashMap<String, Vec<(String, String)>>,
}

impl SymbolGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a directed dependency edge (e.g. from caller to callee, or struct to implemented trait).
    pub fn add_edge(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        kind: impl Into<String>,
    ) {
        let from_str = from.into();
        let to_str = to.into();
        let kind_str = kind.into();

        self.outgoing
            .entry(from_str.clone())
            .or_default()
            .push((to_str.clone(), kind_str.clone()));

        self.incoming
            .entry(to_str)
            .or_default()
            .push((from_str, kind_str));
    }

    /// Multi-hop traversal starting from `root_symbol` up to `max_depth` hops.
    /// Returns a list of visited symbols paired with their hop distance from the root.
    /// Cycle-safe and strictly bounded by `max_depth`.
    pub fn traverse(
        &self,
        root_symbol: &str,
        direction: TraversalDirection,
        max_depth: usize,
    ) -> Vec<(String, usize)> {
        if max_depth == 0 {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut results = Vec::new();

        visited.insert(root_symbol.to_string());
        queue.push_back((root_symbol.to_string(), 0usize));

        while let Some((curr, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let neighbors = match direction {
                TraversalDirection::Outgoing => self.outgoing.get(&curr),
                TraversalDirection::Incoming => self.incoming.get(&curr),
            };

            if let Some(edge_list) = neighbors {
                for (next_node, _) in edge_list {
                    if visited.insert(next_node.clone()) {
                        results.push((next_node.clone(), depth + 1));
                        queue.push_back((next_node.clone(), depth + 1));
                    }
                }
            }
        }

        results
    }

    /// Find the shortest directed path between `start_symbol` and `target_symbol` (BFS).
    /// Returns `Some(vec![start, intermediate..., target])` or `None` if unreachable.
    pub fn find_shortest_path(
        &self,
        start_symbol: &str,
        target_symbol: &str,
    ) -> Option<Vec<String>> {
        if start_symbol == target_symbol {
            return Some(vec![start_symbol.to_string()]);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<String, String> = HashMap::new();

        visited.insert(start_symbol.to_string());
        queue.push_back(start_symbol.to_string());

        let mut found = false;
        while let Some(curr) = queue.pop_front() {
            if curr == target_symbol {
                found = true;
                break;
            }

            if let Some(neighbors) = self.outgoing.get(&curr) {
                for (next_node, _) in neighbors {
                    if visited.insert(next_node.clone()) {
                        parent_map.insert(next_node.clone(), curr.clone());
                        queue.push_back(next_node.clone());
                    }
                }
            }
        }

        if !found {
            return None;
        }

        // Reconstruct path backwards
        let mut path = Vec::new();
        let mut curr = target_symbol.to_string();
        path.push(curr.clone());

        while let Some(prev) = parent_map.get(&curr) {
            path.push(prev.clone());
            curr = prev.clone();
            if curr == start_symbol {
                break;
            }
        }

        path.reverse();
        Some(path)
    }

    /// Compute strongly connected components (SCCs) using Tarjan's linear-time algorithm O(V + E).
    ///
    /// Returns all components. Components with size > 1 (or size 1 with a self-loop) represent cycles.
    pub fn strongly_connected_components(&self) -> Vec<Vec<String>> {
        // Collect all unique node identifiers
        let mut all_nodes = HashSet::new();
        for (u, edges) in &self.outgoing {
            all_nodes.insert(u.clone());
            for (v, _) in edges {
                all_nodes.insert(v.clone());
            }
        }
        for (u, edges) in &self.incoming {
            all_nodes.insert(u.clone());
            for (v, _) in edges {
                all_nodes.insert(v.clone());
            }
        }

        struct TarjanState<'a> {
            graph: &'a HashMap<String, Vec<(String, String)>>,
            index: usize,
            stack: Vec<String>,
            on_stack: HashSet<String>,
            indices: HashMap<String, usize>,
            lowlink: HashMap<String, usize>,
            sccs: Vec<Vec<String>>,
        }

        fn strongconnect(node: &str, state: &mut TarjanState<'_>) {
            state.indices.insert(node.to_string(), state.index);
            state.lowlink.insert(node.to_string(), state.index);
            state.index += 1;
            state.stack.push(node.to_string());
            state.on_stack.insert(node.to_string());

            if let Some(edges) = state.graph.get(node) {
                for (next_node, _) in edges {
                    if !state.indices.contains_key(next_node) {
                        strongconnect(next_node, state);
                        let next_low = state.lowlink[next_node];
                        let curr_low = state.lowlink.get_mut(node).unwrap();
                        if next_low < *curr_low {
                            *curr_low = next_low;
                        }
                    } else if state.on_stack.contains(next_node) {
                        let next_idx = state.indices[next_node];
                        let curr_low = state.lowlink.get_mut(node).unwrap();
                        if next_idx < *curr_low {
                            *curr_low = next_idx;
                        }
                    }
                }
            }

            if state.lowlink[node] == state.indices[node] {
                let mut component = Vec::new();
                while let Some(w) = state.stack.pop() {
                    state.on_stack.remove(&w);
                    component.push(w.clone());
                    if w == node {
                        break;
                    }
                }
                state.sccs.push(component);
            }
        }

        let mut state = TarjanState {
            graph: &self.outgoing,
            index: 0,
            stack: Vec::new(),
            on_stack: HashSet::new(),
            indices: HashMap::new(),
            lowlink: HashMap::new(),
            sccs: Vec::new(),
        };

        for node in all_nodes {
            if !state.indices.contains_key(&node) {
                strongconnect(&node, &mut state);
            }
        }

        state.sccs
    }

    /// Returns `true` if the directed graph contains at least one cycle.
    pub fn has_cycle(&self) -> bool {
        let sccs = self.strongly_connected_components();
        for scc in sccs {
            if scc.len() > 1 {
                return true;
            }
            if scc.len() == 1
                && let Some(edges) = self.outgoing.get(&scc[0])
                && edges.iter().any(|(v, _)| v == &scc[0])
            {
                return true;
            }
        }
        false
    }

    /// Compute a topological sort of the graph nodes if acyclic.
    /// Returns `Some(Vec<String>)` in dependency execution order (sources before sinks),
    /// or `None` if the graph contains cycles (using Kahn's algorithm).
    pub fn topological_sort(&self) -> Option<Vec<String>> {
        if self.has_cycle() {
            return None;
        }

        let mut in_degrees: HashMap<String, usize> = HashMap::new();
        let mut all_nodes = HashSet::new();

        for (u, edges) in &self.outgoing {
            all_nodes.insert(u.clone());
            for (v, _) in edges {
                all_nodes.insert(v.clone());
                *in_degrees.entry(v.clone()).or_insert(0) += 1;
            }
        }
        for u in &all_nodes {
            in_degrees.entry(u.clone()).or_insert(0);
        }

        // Kahn's algorithm
        let mut queue = VecDeque::new();
        for (node, &deg) in &in_degrees {
            if deg == 0 {
                queue.push_back(node.clone());
            }
        }

        let mut order = Vec::new();
        while let Some(u) = queue.pop_front() {
            order.push(u.clone());
            if let Some(neighbors) = self.outgoing.get(&u) {
                for (v, _) in neighbors {
                    if let Some(deg) = in_degrees.get_mut(v) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(v.clone());
                        }
                    }
                }
            }
        }

        if order.len() == all_nodes.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Compute the full transitive closure of reachable symbols from a given start symbol.
    pub fn transitive_closure(&self, start_symbol: &str) -> HashSet<String> {
        let mut closure = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_symbol.to_string());
        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = self.outgoing.get(&curr) {
                for (next_node, _) in neighbors {
                    if closure.insert(next_node.clone()) {
                        queue.push_back(next_node.clone());
                    }
                }
            }
        }

        closure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_hop_traversal_outgoing() {
        let mut graph = SymbolGraph::new();
        // A -> B -> C -> D
        graph.add_edge("fn_A", "fn_B", "calls");
        graph.add_edge("fn_B", "fn_C", "calls");
        graph.add_edge("fn_C", "fn_D", "calls");

        // 1-hop outgoing from A
        let hops_1 = graph.traverse("fn_A", TraversalDirection::Outgoing, 1);
        assert_eq!(hops_1, vec![("fn_B".to_string(), 1)]);

        // 2-hops outgoing from A
        let hops_2 = graph.traverse("fn_A", TraversalDirection::Outgoing, 2);
        assert_eq!(
            hops_2,
            vec![("fn_B".to_string(), 1), ("fn_C".to_string(), 2)]
        );

        // 3-hops outgoing from A
        let hops_3 = graph.traverse("fn_A", TraversalDirection::Outgoing, 3);
        assert_eq!(
            hops_3,
            vec![
                ("fn_B".to_string(), 1),
                ("fn_C".to_string(), 2),
                ("fn_D".to_string(), 3)
            ]
        );
    }

    #[test]
    fn test_multi_hop_traversal_incoming_and_cycles() {
        let mut graph = SymbolGraph::new();
        // Cycle: A -> B -> C -> A
        graph.add_edge("node_A", "node_B", "calls");
        graph.add_edge("node_B", "node_C", "calls");
        graph.add_edge("node_C", "node_A", "calls");

        // Traversal terminates safely despite cycle
        let visited = graph.traverse("node_A", TraversalDirection::Outgoing, 10);
        assert_eq!(visited.len(), 2); // only B and C are discovered as new nodes
    }

    #[test]
    fn test_find_shortest_path() {
        let mut graph = SymbolGraph::new();
        graph.add_edge("main", "init_config", "calls");
        graph.add_edge("main", "start_server", "calls");
        graph.add_edge("start_server", "bind_socket", "calls");
        graph.add_edge("bind_socket", "listen", "calls");

        let path = graph.find_shortest_path("main", "listen").unwrap();
        assert_eq!(path, vec!["main", "start_server", "bind_socket", "listen"]);

        let unreachable = graph.find_shortest_path("init_config", "listen");
        assert!(unreachable.is_none());
    }

    #[test]
    fn test_tarjan_scc_and_cycle_detection() {
        let mut graph = SymbolGraph::new();
        // A -> B -> C -> A (cycle)
        // D -> C (incoming into cycle)
        graph.add_edge("A", "B", "calls");
        graph.add_edge("B", "C", "calls");
        graph.add_edge("C", "A", "calls");
        graph.add_edge("D", "C", "calls");

        assert!(graph.has_cycle());

        let sccs = graph.strongly_connected_components();
        let cycle_scc = sccs
            .iter()
            .find(|c| c.len() > 1)
            .expect("expected cycle SCC");
        assert_eq!(cycle_scc.len(), 3);
        assert!(cycle_scc.contains(&"A".to_string()));
        assert!(cycle_scc.contains(&"B".to_string()));
        assert!(cycle_scc.contains(&"C".to_string()));

        // Acyclic graph
        let mut acyclic = SymbolGraph::new();
        acyclic.add_edge("X", "Y", "calls");
        acyclic.add_edge("Y", "Z", "calls");
        assert!(!acyclic.has_cycle());
    }

    #[test]
    fn test_topological_sort_and_transitive_closure() {
        let mut graph = SymbolGraph::new();
        // Pipeline: parser -> typechecker -> optimizer -> codegen
        graph.add_edge("parser", "typechecker", "calls");
        graph.add_edge("typechecker", "optimizer", "calls");
        graph.add_edge("optimizer", "codegen", "calls");

        let topo = graph.topological_sort().expect("expected valid topo sort");
        let pos_parser = topo.iter().position(|x| x == "parser").unwrap();
        let pos_typecheck = topo.iter().position(|x| x == "typechecker").unwrap();
        let pos_opt = topo.iter().position(|x| x == "optimizer").unwrap();
        let pos_codegen = topo.iter().position(|x| x == "codegen").unwrap();

        assert!(pos_parser < pos_typecheck);
        assert!(pos_typecheck < pos_opt);
        assert!(pos_opt < pos_codegen);

        // Transitive closure from parser includes all 3 downstream
        let closure = graph.transitive_closure("parser");
        assert_eq!(closure.len(), 3);
        assert!(closure.contains("typechecker"));
        assert!(closure.contains("optimizer"));
        assert!(closure.contains("codegen"));

        // From optimizer, only codegen
        let opt_closure = graph.transitive_closure("optimizer");
        assert_eq!(opt_closure.len(), 1);
        assert!(opt_closure.contains("codegen"));
    }
}
