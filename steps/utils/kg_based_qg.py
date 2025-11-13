import json, os, string, nltk, sys
import numpy as np
import networkx as nx
from rapidfuzz import fuzz
from copy import deepcopy
from collections import Counter
from typing import List, Dict, Tuple, Set, Union, Optional


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PORTER_STEMMER = nltk.stem.PorterStemmer()


class KGBasedQGUtilsHelper:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def is_proper_noun_phrase(self, substrings: List[str], full_string: str):
        def get_entity_status(input_str: str):
            entity_status = []
            casing_status = []
            tokens = []
            for i, tok in enumerate(self.nlp(input_str)):
                if tok.text.lower() in ENGLISH_STOPWORDS: continue
                if tok.ent_type_: entity_status.append(True)
                else: entity_status.append(False)

                tokens.append(tok.text)

                if (tok.text.istitle() or tok.text.isupper()) and i > 0: casing_status.append(True)
                else: casing_status.append(False)

            final_entity_status = [es or cs for es, cs in zip(entity_status, casing_status)]
            
            return tokens, final_entity_status
        
        full_string_tokens, full_string_entity_status = get_entity_status(full_string)

        res = []
        for substring in substrings:
            substring_tokens, _ = get_entity_status(substring)

            matched_index = -1
            for i in range(len(full_string_tokens)):
                if full_string_tokens[i: i + len(substring_tokens)] == substring_tokens: 
                    matched_index = i
                    break
            
            if matched_index != -1:
                substring_entity_status = [stt for tok, stt in zip(full_string_tokens[matched_index: matched_index + len(substring_tokens)],
                                                                   full_string_entity_status[matched_index: matched_index + len(substring_tokens)]) if tok.lower() not in ENGLISH_STOPWORDS]
                if sum(substring_entity_status) / len(substring_entity_status) >= 2/3: res.append(True)
                else: res.append(False)

            else: res.append(False)

        return res

class KGBasedQGUtils:
    def __init__(self, 
                 LLM,
                 question_generation_prompt = None,
                 graph_builder_prompt = None,
                 question_refinement_prompt = None):
        self.LLM = LLM
        self.question_generation_prompt = question_generation_prompt
        self.graph_builder_prompt = graph_builder_prompt
        self.question_refinement_prompt = question_refinement_prompt
        self.text_splitter = lambda text: [item.strip(string.punctuation).lower() for item in text.replace("_", " ").replace("-", " ").split()]
        self.helper = KGBasedQGUtilsHelper()


    def _convert_kg_to_nx_graph(self, knowledge_graph: List[Dict[str, str]]) -> Tuple[nx.DiGraph, Set[str]]:
        graph = nx.DiGraph()
        heads = set()
        for edge in knowledge_graph:
            head, tail = edge.get("head"), edge.get("tail")
            if head and tail:
                graph.add_edge(head, tail)
                heads.add(head)
        return graph, heads
    
    def _get_bad_nodes(self, all_nodes: List[str], keypoints: Optional[List[str]] = None, knowledge_graph: Optional[List[Dict[str, str]]] = None):
        # there are several types of nodes that we define as bad. These "bad" nodes will not be masked

        # nodes that overlap with other nodes, because when we mask one, it will still be there in another
        overlapping_nodes = set()
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2 and fuzz.partial_ratio(self.text_splitter(node1), self.text_splitter(node2)) >= 80:
                    overlapping_nodes.add(node1)
                    overlapping_nodes.add(node2)

        # nodes whose component word is not capitalized (excluding stop words)
        # capitalization_status = {node: [int(word.istitle() or word.isupper()) for word in node.split() if word.lower() not in ENGLISH_STOPWORDS] for node in all_nodes}
        # non_capitalized_nodes = {node for node in all_nodes if sum(capitalization_status[node]) / len(capitalization_status[node]) < 2/3}
        is_entity = self.helper.is_proper_noun_phrase(all_nodes, "\n".join(keypoints))
        non_entity_nodes = {node for i, node in enumerate(all_nodes) if not is_entity[i]}

        # # nodes that are single words
        # single_word_nodes = {node for node in all_nodes if len(node.split()) == 1}

        # nodes that do not cover all keypoints
        lack_coverage_nodes = {triplet["head"] for triplet in knowledge_graph if len(triplet["head_coverage"]) != len(keypoints)}

        bad_nodes = overlapping_nodes | non_entity_nodes | lack_coverage_nodes

        if keypoints:
            keypoints_str = "\n".join(keypoints)
            # nodes that are not found in the keypoints
            nodes_not_found_in_keypoints = {node for node in all_nodes if node not in keypoints_str}
            bad_nodes = bad_nodes | nodes_not_found_in_keypoints

        if knowledge_graph is not None:
            # non-exclusive nodes, where an exclusive node is one that is the only head of any relation and also is the only tail of any relation 
            def get_nonexclusive_nodes(knowledge_graph):
                rel_info = {}
                for edge in knowledge_graph:
                    rel = edge.get("relation")
                    head = edge.get("head")
                    tail = edge.get("tail")

                    if rel not in rel_info: rel_info[rel] = {"heads": set(), "tails": set()}
                    rel_info[rel]["heads"].add(head)
                    rel_info[rel]["tails"].add(tail)

                nonexclusive = set()
                for rel in rel_info:
                    heads = rel_info[rel]["heads"]
                    if len(heads) > 1: nonexclusive.update(heads)

                    tails = rel_info[rel]["tails"]
                    if len(tails) > 1: nonexclusive.update(tails)

                return nonexclusive

            nonexclusive_nodes = get_nonexclusive_nodes(knowledge_graph)
            bad_nodes = bad_nodes | nonexclusive_nodes

        return bad_nodes
    

    def _knowledge_graph_masking(self, 
                                 knowledge_graph: List[Dict[str, str]], 
                                 max_nodes_to_mask: int,
                                 keypoints: List[str]):
        graph, heads = self._convert_kg_to_nx_graph(knowledge_graph)

        bad_nodes = self._get_bad_nodes(list(graph.nodes()), keypoints, knowledge_graph)

        if keypoints: keypoints_str = "\n".join(keypoints)
        else: keypoints_str = ""
        
        traversal_order = None
        for i in range(len(knowledge_graph)):
            if knowledge_graph[i].get("head") and knowledge_graph[i].get("head") not in bad_nodes:
                traversal_order = list(range(i, len(knowledge_graph))) + list(range(0, i))
                break
        
        if traversal_order is None: return []
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
                # since the masked node is going to be the head of the relation, there's no need to check if this node is leaf
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
    

    def kg_postprocessing(self, knowledge_graph: List[Dict[str, str]], keypoints: List[str]) -> List[Dict[str, str]]:

        def get_original_string(input_str, keypoints_str):
            input_str_lower = input_str.lower()
            keypoints_str_lower = keypoints_str.lower()

            try: 
                start_index = keypoints_str_lower.index(input_str_lower)
            except ValueError: 
                return input_str

            return keypoints_str[start_index: start_index + len(input_str_lower)]
        

        keypoints_str = "\n".join(keypoints)
        postprocessed_knowledge_graph = deepcopy(knowledge_graph)
        for i, rel in enumerate(knowledge_graph):
            head = rel["head"]
            tail = rel["tail"]

            postprocessed_head = get_original_string(head, keypoints_str)
            postprocessed_tail = get_original_string(tail, keypoints_str)

            head_coverage = [i for i, kp in enumerate(keypoints) if postprocessed_head in kp]
            tail_coverage = [i for i, kp in enumerate(keypoints) if postprocessed_tail in kp]

            postprocessed_knowledge_graph[i]["head"] = postprocessed_head
            postprocessed_knowledge_graph[i]["tail"] = postprocessed_tail
            postprocessed_knowledge_graph[i]["head_coverage"] = head_coverage
            postprocessed_knowledge_graph[i]["tail_coverage"] = tail_coverage

        return postprocessed_knowledge_graph



    
    def extract_kg_from_keypoints(self, keypoints: List[str]) -> List[Dict[str, str]]:
        concatenated_keypoints = "\n".join(["- " + item for item in keypoints])
        user_prompt = self.graph_builder_prompt["user"][:]\
            .replace("[ADD_KEYPOINTS_HERE]", concatenated_keypoints)
        
        _result = self.LLM.generate(
            system_prompt = self.graph_builder_prompt["system"],
            user_prompt = user_prompt
        ).replace("Knowledge Graph:", "").strip()

        if _result.startswith("```json"):
            _result = _result[7:-3]

        try:
            return self.kg_postprocessing(knowledge_graph = json.loads(_result), keypoints = keypoints)
        except Exception as e:
            return {"error": str(e)}
    

    def generate_question_from_masked_kg_step_by_step(self, masked_kg: List[Dict[str, str]], masked_keypoints_str: Optional[str] = None) -> List[str]:
        def format_user_prompt(generated_questions: List[str], relation_texts: List[str], question_target_type: str):
            assert len(generated_questions) == len(relation_texts)
            
            res = []
            for relation_text, generated_question in zip(relation_texts, generated_questions[1:] + [""]):
                to_append = f"Relation: {relation_text}\nGenerated question: {generated_question}"
                res.append(to_append)

            res = "\n\n".join(res)
            user_prompt = f"""User input:\nThe generated question must ask about an a/an '{question_target_type}'\nQuestion generation steps:\n{res}"""
            return user_prompt

        question_target_type = masked_kg[0]["head_type"]
        generated_questions = [""]
        relation_texts = []
        for i, rel in enumerate(masked_kg):

            head_str = rel["head"]
            tail_str = rel["tail"]
            relation_text = f"{head_str} [{rel['head_type']}] | {rel['relation']} | {tail_str} [{rel['tail_type']}]"
            relation_texts.append(relation_text)

            current_question = generated_questions[-1]

            user_prompt = format_user_prompt(generated_questions, relation_texts, question_target_type)

            resp = self.LLM.client.chat.completions.create(
                model=self.LLM.model_name,
                messages=[
                    {"role": "system", "content": self.question_generation_prompt["system"]},
                    {"role": "user", "content": user_prompt}
                ],
                prediction={
                    "type": "content",
                    "content": current_question
                },
                max_tokens = 128,
            )
            question = resp.choices[0].message.content.strip()

            print(question)

            generated_questions.append(question)

        return generated_questions[1:]
    

    def question_refinement(self, generated_question: str, masked_keypoints_str: str, masked_kg: List[Dict[str, str]]):
        question_target_type = masked_kg[0]["head_type"]

        user_prompt = self.question_refinement_prompt["user"][:]\
        .replace("[ADD_KEYPOINTS_HERE]", masked_keypoints_str)\
        .replace("[ADD_QUESTION_HERE]", generated_question)\
        .replace("[ADD_QUESTION_TARGET_TYPE]", question_target_type)

        res = self.LLM.generate(
            system_prompt = self.question_refinement_prompt["system"],
            user_prompt = user_prompt,
            max_output_tokens = 128
        ).strip()

        return res




class KGBasedQGChecker:
    def __init__(self, 
                 minicheck_model_name, 
                 minicheck_cache_dir, 
                 question_answerability_check_prompt = None,
                 LLM = None):
        from minicheck.minicheck import MiniCheck
        
        if minicheck_cache_dir and minicheck_model_name:
            self.minicheck_scorer = MiniCheck(model_name=minicheck_model_name, cache_dir=minicheck_cache_dir)
        else: self.minicheck_scorer = None
        
        self.stemmer = nltk.stem.PorterStemmer()

        self.LLM = LLM
        self.question_answerability_check_prompt = question_answerability_check_prompt

        self.text_splitter = lambda text: [item.strip(string.punctuation).lower() for item in text.replace("_", " ").replace("-", " ").split()]

    def _check_groundedness_of_relations(self, base_doc: str, relations: List[str], verbose: bool = False) -> bool:
        if self.minicheck_scorer:
            pred_label, _, _, _ = self.minicheck_scorer.score(docs=[base_doc] * len(relations), claims=relations)
        else:
            pred_label = [1] * len(relations)

        if verbose: print(pred_label)
        return all(pred_label)
    
    def _check_word_based_completeness(self, base_doc: str, relations: List[str], threshold: float = 0.75) -> bool:
        base_doc_words = set([self.stemmer.stem(w) for w in self.text_splitter(base_doc) if w not in ENGLISH_STOPWORDS])

        relations_words = set([])
        for rel in relations:
            rel_words = set([self.stemmer.stem(w) for w in self.text_splitter(rel) if w not in ENGLISH_STOPWORDS])
            relations_words.update(rel_words)

        check = len(relations_words.intersection(base_doc_words)) / len(base_doc_words)
        return check >= threshold
    
    def _max_relation_occurrences(self, knowledge_graph: List[Dict[str, str]]) -> int:
        relcounter = Counter()
        for edge in knowledge_graph:
            relation = edge["relation"]
            relcounter[relation] += 1

        return max(relcounter.values())

    def _check_graph_connectedness(self, knowledge_graph):
        graph = nx.DiGraph()
        if not graph: return False
        
        for edge in knowledge_graph:
            head, tail = edge.get("head"), edge.get("tail")
            if head and tail:
                graph.add_edge(head, tail)
        return nx.is_weakly_connected(graph)

    def check_completeness_of_extracted_kg(self, knowledge_graph: List[Dict[str, str]], keypoints: List[str], word_check_threshold: float = 0.75, verbose: bool = False) -> bool:
        # connectedness_check = self._check_graph_connectedness(knowledge_graph)
        # if not connectedness_check: return False

        keypoints_str = "\n".join(keypoints)

        relations = [f"{rel['head']} {rel['relation']} {rel['tail']}" for rel in knowledge_graph]

        word_based_completeness = self._check_word_based_completeness(base_doc = keypoints_str, relations = relations, threshold = word_check_threshold)
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

        if self.minicheck_scorer:
            pred_label, _, _, _ = self.minicheck_scorer.score(docs=[p[0] for p in pairs], claims=[p[1] for p in pairs])
        else:
            pred_label = [1] * len(pairs)

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

            if self.minicheck_scorer:
                pred_label, _, _, _ = self.minicheck_scorer.score(docs=[question] * len(relations_to_check), claims=relations_to_check)
            else:
                pred_label = [1] * len(relations_to_check)

            for i, l in enumerate(pred_label):
                if not l:
                    error_counter[i] += 1
            
        return all([v <= 1 for v in error_counter.values()])
                
    def _check_question_progression_via_word_counting(self, generated_questions: List[str], threshold: float = 0.8):
        generated_questions_words = [set(self.text_splitter(q)) for q in generated_questions]

        for i in range(1, len(generated_questions_words)):
            current = generated_questions_words[i]
            previous = generated_questions_words[i-1]

            intersection = current.intersection(previous)
            percentage = len(intersection)/ len(previous)

            if percentage < threshold: return False

        return True

            

    def check_correctness_of_question_progression(self, generated_questions: List[str], masked_knowledge_graph: List[Dict[str, str]], word_counting_check_threshold: float = 0.8) -> bool:
        if not len(generated_questions) == len(masked_knowledge_graph): return False

        check_word_counting_based = self._check_question_progression_via_word_counting(generated_questions, threshold = word_counting_check_threshold)
        if not check_word_counting_based: return False

        relation_question_correspondence_nli = self._check_relation_question_correspondence(generated_questions, masked_knowledge_graph) # length
        return relation_question_correspondence_nli
    

    def check_question_answerability(self, question: str, verbose: bool = False):

        user_prompt = self.question_answerability_check_prompt["user"][:]\
        .replace("[ADD_QUESTION_HERE]", question)

        res = self.LLM.generate(
            system_prompt = self.question_answerability_check_prompt["system"],
            user_prompt = user_prompt,
            max_output_tokens = 16
        ).strip()
        
        if verbose: print(res)

        return "B." in res
    
