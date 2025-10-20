import json, os, string, nltk, sys
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Union, Optional


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PORTER_STEMMER = nltk.stem.PorterStemmer()


class KGBasedQGUtils:
    def __init__(self, openai_client,
                 question_generation_prompt,
                 graph_builder_prompt):
        self.openai_client = openai_client
        self.question_generation_prompt = question_generation_prompt
        self.graph_builder_prompt = graph_builder_prompt


    def _convert_kg_to_nx_graph(self, knowledge_graph: List[Dict[str, str]]) -> Tuple[nx.DiGraph, Set[str]]:
        graph = nx.DiGraph()
        heads = set()
        for edge in knowledge_graph:
            head, tail = edge.get("head"), edge.get("tail")
            if head and tail:
                graph.add_edge(head, tail)
                heads.add(head)
        return graph, heads
    
    def _get_overlapping_nodes(self, all_nodes: List[str]):
        overlapping_nodes = set()
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2 and node1 in node2:
                    overlapping_nodes.add(node1)
                    overlapping_nodes.add(node2)

        return overlapping_nodes

    def _find_starting_node(self, graph: nx.DiGraph, heads: Set[str]) -> Optional[str]:
        degrees = graph.degree
        all_nodes = list(graph.nodes())

        # Candidates cannot include overlapping nodes
        overlapping_nodes = self._get_overlapping_nodes(all_nodes)

        # Find nodes that are not capitalized (using .istitle() as in the original).
        non_capitalized_nodes = {node for node in graph.nodes() if not any([word.istitle() for word in node.split()])}

        # Candidates are the original head nodes, excluding the filtered sets.
        candidates = heads - non_capitalized_nodes - overlapping_nodes
        
        if not candidates:
            return None
        
        # Find the candidate with the highest degree. The key function looks up the
        # degree of each candidate node in the `degrees` view object.
        highest_degree_node = max(candidates, key=lambda node: degrees[node])
        return highest_degree_node



    def _get_traversal_order(self, knowledge_graph: List[Dict[str, str]]) -> List[int]:
        if not knowledge_graph:
            return []

        graph, heads = self._convert_kg_to_nx_graph(knowledge_graph)
        start_node = self._find_starting_node(graph, heads)

        if start_node is None:
            return []

        # Pre-calculate all-pairs shortest path lengths for efficient lookups.
        distances = {source: targets for source, targets in nx.all_pairs_shortest_path_length(graph)}

        # Find the first edge that originates from our chosen start_node.
        try:
            start_edge_index = next(
                i for i, edge in enumerate(knowledge_graph) if edge.get("head") == start_node
            )
        except StopIteration:
            # Fallback for rare cases where the start_node has no outgoing edges.
            return []

        ranked_order = []
        unvisited_indices = set(range(len(knowledge_graph)))
        
        # Initialize the traversal with the starting edge.
        current_edge_index = start_edge_index
        ranked_order.append(current_edge_index)
        unvisited_indices.remove(current_edge_index)

        # Iteratively find the next closest edge until all have been visited.
        while unvisited_indices:
            current_tail = knowledge_graph[current_edge_index]['tail']
            current_head = knowledge_graph[current_edge_index]['head']
            
            min_dist = float('inf')
            next_edge_index = -1

            # Find the unvisited edge whose head is closest to the current edge's tail.
            for candidate_index in unvisited_indices:
                candidate_head = knowledge_graph[candidate_index]['head']
                candidate_tail = knowledge_graph[candidate_index]["tail"]
                
                # Look up the path distance; default to infinity if no path exists.
                dist = min(distances.get(current_tail, {}).get(candidate_head, float('inf')),
                           distances.get(current_head, {}).get(candidate_head, float('inf')),
                           distances.get(current_tail, {}).get(candidate_tail, float('inf')),
                           distances.get(current_head, {}).get(candidate_tail, float('inf')),)

                if dist < min_dist:
                    min_dist = dist
                    next_edge_index = candidate_index
            
            # If the remaining edges are in a disconnected component of the graph.
            if next_edge_index == -1:
                # Fallback: simply pick an arbitrary unvisited edge to continue.
                next_edge_index = unvisited_indices.pop()
            else:
                unvisited_indices.remove(next_edge_index)
            
            ranked_order.append(next_edge_index)
            current_edge_index = next_edge_index

        return ranked_order
    

    def _knowledge_graph_masking(self, knowledge_graph: List[Dict[str, str]], traversal_order: List[int], max_nodes_to_mask: int) -> List[Dict[str, str]]:
        graph, heads = self._convert_kg_to_nx_graph(knowledge_graph)
        overlapping_nodes = self._get_overlapping_nodes(list(graph.nodes()))

        max_nodes_to_mask = min(len(graph.nodes()) - 1, max_nodes_to_mask)

        first_edge = knowledge_graph[traversal_order[0]]

        masked_entities_info = [[first_edge["head"], first_edge["head_type"]]]
        masked_entities_names = [first_edge["head"]]


        for j in range(1, len(traversal_order)):
            i = traversal_order[j]
            relation = knowledge_graph[i]
            is_jump = not graph.has_edge(knowledge_graph[j-1]["tail"], relation["head"])

            if all([
                len(masked_entities_names) < max_nodes_to_mask, # number of masked nodes has not exceed limit
                not is_jump, # current head is the tail of the previous relation
                all([masked_entities_info[k][1] != relation["head_type"] for k in range(len(masked_entities_info))]), # a node with the same type as head has not been masked
                not relation["head"] in masked_entities_names, # the head itself has not been masked
                not relation["head"] in overlapping_nodes, # the head is not a node that overlap with other nodes
            ]):
                # do the masking of the current head
                masked_entities_info.append([relation["head"], relation["head_type"]])
                masked_entities_names.append(relation["head"])
        
        masked_kg = []
        for i in traversal_order:
            relation = knowledge_graph[i]
            masked_kg.append({
                "head": relation["head"] if relation["head"] not in masked_entities_names else f"<Unknown> #{masked_entities_names.index(relation['head']) + 1}",
                "head_unmasked": relation["head"],
                "head_type": relation["head_type"],
                "relation": relation["relation"],
                "tail": relation["tail"] if relation["tail"] not in masked_entities_names else f"<Unknown> #{masked_entities_names.index(relation['tail']) + 1}",
                "tail_unmasked": relation["tail"],
                "tail_type": relation["tail_type"]
            })

        return masked_kg, len(masked_entities_names)
    

    def _build_user_prompt_from_masked_kg(self, masked_kg: List[Dict[str, str]]) -> str:
        prompt = "Relations: (subject [subject type] | relation | object [object type])" # f"""**Full text:**\n{text}\n\n**Relations:**"""
        for i, rel in enumerate(masked_kg):

            head_str = rel["head"]
            tail_str = rel["tail"]
            relation_text = f"{head_str} [{rel['head_type']}] | {rel['relation']} | {tail_str} [{rel['tail_type']}]"

            prompt += f"\n{i+1}. {relation_text}"

        return prompt + f"\n\nQuestion generation steps: ({len(masked_kg)} steps)"


    
    def extract_kg_from_keypoints(self, keypoints: List[str]) -> List[Dict[str, str]]:
        concatenated_keypoints = "\n".join(["- " + item for item in keypoints])
        user_prompt = self.graph_builder_prompt["user"][:]\
            .replace("[ADD_KEYPOINTS_HERE]", concatenated_keypoints)
        
        resp = self.openai_client["client"].chat.completions.create(
            model=self.openai_client["model"],
            messages=[
                {
                    "role": "system",
                    "content": self.graph_builder_prompt["system"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 1024,
        )

        _result = resp.choices[0].message.content.replace("Knowledge Graph:", "").strip()

        if _result.startswith("```json"):
            _result = _result[7:-3]

        try:
            return json.loads(_result)
        except Exception as e:
            return {"error": str(e)}
        

    def generate_question_from_masked_kg(self, masked_kg: List[Dict[str, str]]) -> List[str]:
        user_prompt = self._build_user_prompt_from_masked_kg(masked_kg)

        resp = self.openai_client["client"].chat.completions.create(
            model=self.openai_client["model"],
            messages=[
                {
                    "role": "system",
                    "content": self.question_generation_prompt["system"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 1024,
        )

        _generated_questions = resp.choices[0].message.content.strip()

        if "<Unknown>" in _generated_questions: return []

        generated_questions = [question[3:] for question in _generated_questions.split("\n")]

        return generated_questions
    



class KGBasedQGChecker:
    def __init__(self, 
                 minicheck_model_name, 
                 minicheck_cache_dir, 
                 openai_client = None):
        from minicheck.minicheck import MiniCheck
        
        if minicheck_cache_dir and minicheck_model_name:
            self.minicheck_scorer = MiniCheck(model_name=minicheck_model_name, cache_dir=minicheck_cache_dir)
        else: self.minicheck_scorer = None
        
        self.stemmer = nltk.stem.PorterStemmer()

        self.openai_client = openai_client

        self.text_splitter = lambda text: [item.strip(string.punctuation).lower() for item in text.replace("_", " ").replace("-", " ").split()]

    def _check_groundedness_of_relations(self, base_doc: str, relations: List[str], verbose: bool = False) -> bool:
        pred_label, _, _, _ = self.minicheck_scorer.score(docs=[base_doc] * len(relations), claims=relations)

        if verbose: print(pred_label)
        return all(pred_label)
    
    def _check_word_based_completeness(self, base_doc: str, relations: List[str]) -> bool:
        base_doc_words = set([self.stemmer.stem(w) for w in self.text_splitter(base_doc) if w not in ENGLISH_STOPWORDS])

        relations_words = set([])
        for rel in relations:
            rel_words = set([self.stemmer.stem(w) for w in self.text_splitter(rel) if w not in ENGLISH_STOPWORDS])
            relations_words.update(rel_words)

        check = len(relations_words.intersection(base_doc_words)) / len(base_doc_words)
        return check >= 0.75

    def _check_graph_connectedness(self, knowledge_graph):
        graph = nx.DiGraph()
        for edge in knowledge_graph:
            head, tail = edge.get("head"), edge.get("tail")
            if head and tail:
                graph.add_edge(head, tail)
        return nx.is_weakly_connected(graph)

    def check_completeness_of_extracted_kg(self, knowledge_graph: List[Dict[str, str]], keypoints: List[str], verbose: bool = False) -> bool:
        connectedness_check = self._check_graph_connectedness(knowledge_graph)
        if not connectedness_check: return False

        keypoints_str = "\n".join(keypoints)

        relations = [f"{rel['head']} {rel['relation']} {rel['tail']}" for rel in knowledge_graph]

        word_based_completeness = self._check_word_based_completeness(base_doc = keypoints_str, relations = relations)
        if not word_based_completeness: 
            if verbose:
                print("Failed word-based completeness check:", word_based_completeness)
            return False

        relations_groundedness = self._check_groundedness_of_relations(base_doc = keypoints_str, relations = relations, verbose = verbose)

        return relations_groundedness
    

    def _check_consistency_question_pairs(self, generated_questions: List[str]) -> List[bool]:
        # function to check if current question contain previous one

        pairs = []
        for i in range(1, len(generated_questions)):
            current_question = generated_questions[i]
            previous_question = generated_questions[i - 1]

            pairs.append([current_question, previous_question])

        pred_label, _, _, _ = self.minicheck_scorer.score(docs=[p[0] for p in pairs], claims=[p[1] for p in pairs])

        return pred_label


    def _check_relation_question_correspondence(self, generated_questions: List[str], masked_knowledge_graph: List[Dict[str, str]]) -> List[bool]:
        relations = []
        for rel in masked_knowledge_graph:
            head_masked = "<Unknown>" in rel['head']
            tail_masked = "<Unknown>" in rel['tail']

            if head_masked and tail_masked: 
                to_append = None
            else:
                head_str = rel['head'] if not head_masked else f"Some {rel['head_type']}"
                tail_str = rel['tail'] if not tail_masked else f"Some {rel['tail_type']}"
                to_append = f"{head_str} {rel['relation']} {tail_str}"

            relations.append(to_append)

        for i in range(len(generated_questions)):
            question = generated_questions[i]
            relations_to_check = relations[:i + 1]
            relations_to_check = [rel for rel in relations_to_check if rel]
            if not relations_to_check: continue

            pred_label, _, _, _ = self.minicheck_scorer.score(docs=[question] * len(relations_to_check), claims=relations_to_check)

            if not all(pred_label): return False

        return True
                

            

    def check_correctness_of_question_progression(self, generated_questions: List[str], masked_knowledge_graph: List[Dict[str, str]]) -> bool:
        if not len(generated_questions) == len(masked_knowledge_graph): return False

        return self._check_relation_question_correspondence(generated_questions, masked_knowledge_graph) # length
    

    def check_question_answerability(self, question: str):

        resp = self.openai_client["client"].chat.completions.create(
            model=self.openai_client["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant"
                },
                {
                    "role": "user",
                    "content": f"Will this question have multiple or a single answer (How many entity/object can be used to answer the question). \
                        Answer without explaining further\nQuestion: '{question}'\nA. Multiple\nB. Single"
                }
            ],
            # temperature=0.1,
            max_tokens = 16,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )



        res = resp.choices[0].message.content.strip()

        return "B." in res