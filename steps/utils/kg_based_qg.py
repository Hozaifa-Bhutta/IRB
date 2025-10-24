import json, os, string, nltk, sys
import numpy as np
import networkx as nx
from collections import Counter
from typing import List, Dict, Tuple, Set, Union, Optional
from sentence_transformers import SentenceTransformer
from utils.prompts import QUESTION_ANSWERABILITY_CHECK_PROMPT
from pydantic import BaseModel


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PORTER_STEMMER = nltk.stem.PorterStemmer()

class RelationFormat(BaseModel):
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str

class KnowledgeGraphFormat(BaseModel):
    relations: list[RelationFormat]




class KGBasedQGUtils:
    def __init__(self, openai_client,
                 question_generation_prompt = None,
                 graph_builder_prompt = None,
                 question_refinement_prompt = None):
        self.openai_client = openai_client
        self.question_generation_prompt = question_generation_prompt
        self.graph_builder_prompt = graph_builder_prompt
        self.question_refinement_prompt = question_refinement_prompt


    def _convert_kg_to_nx_graph(self, knowledge_graph: List[Dict[str, str]]) -> Tuple[nx.DiGraph, Set[str]]:
        graph = nx.DiGraph()
        heads = set()
        for edge in knowledge_graph:
            head, tail = edge.get("head"), edge.get("tail")
            if head and tail:
                graph.add_edge(head, tail)
                heads.add(head)
        return graph, heads
    
    def _get_bad_nodes(self, all_nodes: List[str], keypoints: Optional[List[str]] = None):
        overlapping_nodes = set()
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2 and node1 in node2:
                    overlapping_nodes.add(node1)
                    overlapping_nodes.add(node2)

        non_capitalized_nodes = {node for node in all_nodes if not any([word.istitle() for word in node.split()])}
        single_word_nodes = {node for node in all_nodes if len(node.split()) == 1}
        bad_nodes = overlapping_nodes | non_capitalized_nodes | single_word_nodes

        if keypoints:
            keypoints_str = "\n".join(keypoints)
            nodes_not_found_in_keypoints = {node for node in all_nodes if node not in keypoints_str}
            bad_nodes = bad_nodes | nodes_not_found_in_keypoints

        return bad_nodes


    def _get_semantic_based_traversal_order(self, knowledge_graph: List[Dict[str, str]], keypoints: List[str]) -> List[int]:
        if not knowledge_graph:
            return []
        
        if not hasattr(self, "text_embedding"):
            self.text_embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        keypoints_str = "\n".join(keypoints)
        relations = [f"{rel['head']} {rel['relation']} {rel['tail']}" for rel in knowledge_graph]

        embeddings = self.text_embedding.encode([keypoints_str] + relations, normalize_embeddings=True)

        keypoints_embeddings = embeddings[0:1]
        relations_embeddings = embeddings[1:]

        keypoints_relations_scores = keypoints_embeddings.dot(relations_embeddings.T)[0]
        traversal_order = np.argsort(-1 * keypoints_relations_scores).tolist()

        return traversal_order
    

    def _knowledge_graph_masking(self, 
                                 knowledge_graph: List[Dict[str, str]], 
                                 traversal_order: List[int], 
                                 max_nodes_to_mask: int,
                                 keypoints: Optional[List[str]] = None):
        graph, heads = self._convert_kg_to_nx_graph(knowledge_graph)

        bad_nodes = self._get_bad_nodes(list(graph.nodes()), keypoints)

        if keypoints: keypoints_str = "\n".join(keypoints)
        else: keypoints_str = ""

        first_edge = knowledge_graph[traversal_order[0]]

        masked_entities_info = [[first_edge["head"], first_edge["head_type"]]]
        masked_entities_names = [first_edge["head"]]


        for j in range(1, len(traversal_order)):
            i = traversal_order[j]
            relation = knowledge_graph[i]
            is_jump = all([knowledge_graph[j-k]["tail"] != relation["head"] for k in range(1, j + 1)])

            if all([
                len(masked_entities_names) < max_nodes_to_mask, # number of masked nodes has not exceed limit
                not is_jump, # current head is the tail of the previous relation
                all([masked_entities_info[k][1] != relation["head_type"] for k in range(len(masked_entities_info))]), # a node with the same type as head has not been masked
                relation["head"] not in masked_entities_names, # the head itself has not been masked
                relation["head"] not in bad_nodes, # self explanatory
            ]):
                # do the masking of the current head
                masked_entities_info.append([relation["head"], relation["head_type"]])
                masked_entities_names.append(relation["head"])
        
        res = []
        for j in range(len(masked_entities_names)):
            num_hops = len(masked_entities_names[:j + 1])
            masked_kg = []
            for i in traversal_order:
                relation = knowledge_graph[i]
                masked_kg.append({
                    "head": relation["head"] if relation["head"] not in masked_entities_names[:j + 1] else f"<Unknown> #{masked_entities_names.index(relation['head']) + 1}",
                    "head_unmasked": relation["head"],
                    "head_type": relation["head_type"],
                    "relation": relation["relation"],
                    "tail": relation["tail"] if relation["tail"] not in masked_entities_names[:j + 1] else f"<Unknown> #{masked_entities_names.index(relation['tail']) + 1}",
                    "tail_unmasked": relation["tail"],
                    "tail_type": relation["tail_type"]
                })

            masked_keypoints_str = keypoints_str[:]
            for ent_name, ent_type in masked_entities_info[:j + 1]:
                masked_keypoints_str = masked_keypoints_str.replace(ent_name, f"<Unknown #{masked_entities_names.index(ent_name) + 1} ({ent_type})>")

            res.append({"masked_kg": masked_kg, "num_hops": num_hops, "masked_keypoints_str": masked_keypoints_str})

        return res
    

    def _build_user_prompt_from_masked_kg(self, masked_kg: List[Dict[str, str]], masked_keypoints_str: Optional[str] = None) -> str:
        prompt = "Relations: (subject [subject type] | relation | object [object type])" # f"""**Full text:**\n{text}\n\n**Relations:**"""
        if masked_keypoints_str:
            prompt = f"Text: {masked_keypoints_str}\n" + prompt


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
    

    def generate_question_from_masked_kg_step_by_step(self, masked_kg: List[Dict[str, str]], masked_keypoints_str: Optional[str] = None) -> List[str]:
        generated_questions = [""]
        for i, rel in enumerate(masked_kg):

            head_str = rel["head"]
            tail_str = rel["tail"]
            relation_text = f"{head_str} [{rel['head_type']}] | {rel['relation']} | {tail_str} [{rel['tail_type']}]"

            current_question = generated_questions[-1]

            user_prompt = f"Relation: {relation_text}"
            if current_question: user_prompt += f"\nExisting question: {current_question}"

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
                prediction={
                    "type": "content",
                    "content": current_question
                },
                max_tokens = 128,
            )

            question = resp.choices[0].message.content.strip()

            generated_questions.append(question)

        return generated_questions[1:]
    

    def question_refinement(self, generated_question: str, keypoints: List[str]):
        keypoints_str = "\n".join(keypoints)

        user_prompt = self.question_refinement_prompt["user"][:]\
        .replace("[ADD_KEYPOINTS_HERE]", keypoints_str)\
        .replace("[ADD_QUESTION_HERE]", generated_question)
        
        resp = self.openai_client["client"].chat.completions.create(
            model=self.openai_client["model"],
            messages=[
                {
                    "role": "system",
                    "content": self.question_refinement_prompt["system"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 128,
        )

        return resp.choices[0].message.content.strip()




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

        error_counter = Counter()
        for i in range(len(generated_questions)):
            question = generated_questions[i]
            relations_to_check = relations[:i + 1]
            relations_to_check = [rel for rel in relations_to_check if rel]
            if not relations_to_check: continue

            pred_label, raw_probs, _, _ = self.minicheck_scorer.score(docs=[question] * len(relations_to_check), claims=relations_to_check)

            for i, l in enumerate(pred_label):
                if not l:
                    error_counter[i] += 1
            
        return all([v <= 1 for v in error_counter.values()])
                

            

    def check_correctness_of_question_progression(self, generated_questions: List[str], masked_knowledge_graph: List[Dict[str, str]]) -> bool:
        if not len(generated_questions) == len(masked_knowledge_graph): return False

        return self._check_relation_question_correspondence(generated_questions, masked_knowledge_graph) # length
    

    def check_question_answerability(self, question: str):

        user_prompt = QUESTION_ANSWERABILITY_CHECK_PROMPT["user"][:]\
        .replace("[ADD_QUESTION_HERE]", question)
        resp = self.openai_client["client"].chat.completions.create(
            model=self.openai_client["model"],
            messages=[
                {
                    "role": "system",
                    "content": QUESTION_ANSWERABILITY_CHECK_PROMPT["system"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            # temperature=0.1,
            max_tokens = 16,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )



        res = resp.choices[0].message.content.strip()

        return "B." in res
    
