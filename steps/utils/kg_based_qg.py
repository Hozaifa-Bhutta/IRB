import json, os, string, nltk, sys, random, dateutil, humanize, pycountry, names, heapq, re, num2words, concurrent.futures, math
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
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

    def is_proper_noun_phrase(self, substrings: List[str], full_string: str, check_threshold: float = 0.75):
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
                if substring_entity_status and sum(substring_entity_status) / len(substring_entity_status) >= check_threshold: res.append(True)
                else: res.append(False)

            else: res.append(False)

        return res
    

    def _knowledge_graph_paraphrase_date(self, 
                                         masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         wikidump_date: str,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def validate_date(date_string):
            date_string = date_string.strip()

            allowed_formats = [
                "%Y-%m-%d",       # 2025-01-28
                "%Y-%m",          # 2025-01
                "%B %d, %Y",      # January 28, 2025
                "%d %B %Y",       # 28 January 2025
                "%B %Y",          # January 2025
                "%b %d, %Y",      # Jan 28, 2025
                "%d %b %Y",       # 28 Jan 2025
                "%Y/%m/%d",       # 2025/01/28
            ]
            for fmt in allowed_formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    return True
                except ValueError:
                    continue
            return False

        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Date":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Date":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if validate_date(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        wikidump_dt = datetime.strptime(wikidump_date, "%Y-%m-%d")

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:

            try:
                dt = dateutil.parser.parse(node_value,)
                td = wikidump_dt - dt
                if create_false_premise:
                    td = td + timedelta(days = random.choice(range(366, 365 * 5)))

                if td.days == 0:
                    paraphrased_node_value = "today"
                elif abs(td.days) < 7:
                    paraphrased_node_value = "a few days" + (" ago" if td.days > 0 else " from now")
                elif abs(td.days) < 30:
                    paraphrased_node_value = "a few weeks" + (" ago" if td.days > 0 else " from now")
                else:
                    paraphrased_node_value = "roughly " + humanize.naturaltime(td)
            except Exception: continue

            paraphrase_mapper[node_value] = paraphrased_node_value


        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_year(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         wikidump_date: str,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def is_valid_year_datetime(year_str):
            try:
                datetime.strptime(year_str, '%Y')
                return True
            except ValueError:
                return False
            
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Year":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Year":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = [int(node) for node in nodes_to_paraphrase if is_valid_year_datetime(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        wikidump_year = int(datetime.strptime(wikidump_date, "%Y-%m-%d").year)

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:

            year_delta = wikidump_year - node_value
            if create_false_premise:
                year_delta += random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            abs_year_delta = abs(year_delta)
            if year_delta == 0: paraphrased_node_value = "this year"
            elif year_delta > 0: 
                paraphrased_node_value = f"{abs_year_delta} year ago" if abs_year_delta == 1 else f"{abs_year_delta} years ago"
            elif year_delta < 0: 
                paraphrased_node_value = f"{abs_year_delta} year from now" if abs_year_delta == 1 else f"{abs_year_delta} years from now"

            paraphrase_mapper[str(node_value)] = paraphrased_node_value


        return paraphrase_mapper
    
    def _knowledge_graph_paraphrase_person(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):
        def is_valid_person_name(text):
            clean_text = text.strip()
            
            words = clean_text.split()
            
            if len(words) <= 1:
                return False
                
            for word in words:
                if not word[0].isupper():
                    return False
            return True
        
        def abbreviate_name(name, create_false_premise = False):
            words = name.strip().split()
            if not words:
                return ""
            
            if create_false_premise:
                words = words[:-1] + [names.get_last_name()] # replace last name with a random last name
            
            initials = [word[0].upper() + "." for word in words[:-1]]
            
            return " ".join(initials + [words[-1]])
        
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Person":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Person":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if is_valid_person_name(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase


        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:
            paraphrased_node_value = abbreviate_name(node_value, create_false_premise)
            paraphrase_mapper[node_value] = paraphrased_node_value

        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_country(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def country_flag_search(query):
            try:
                search_results = pycountry.countries.search_fuzzy(query)
            except Exception: return None

            if not search_results: return None
            flag = search_results[0].flag

            if (hasattr(search_results[0], "official_name") and query == search_results[0].official_name) or \
                (hasattr(search_results[0], "name") and query == search_results[0].name):
                return flag
            
            return None

        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Country":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Country":
                nodes_to_paraphrase.append(triplet["tail"])
        
        country2flags = {country: country_flag_search(country) for country in nodes_to_paraphrase}
        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if country2flags.get(node) is not None]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:
            if not create_false_premise:
                paraphrased_node_value = "the country whose flag is " + country2flags[node_value]
            else:
                all_countries = list(pycountry.countries)
                for _ in range(10):
                    random_country = random.choice(all_countries).name
                    if random_country != node_value: break
                paraphrased_node_value = random_country
            paraphrase_mapper[node_value] = paraphrased_node_value

        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_quantity_statistics_number(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):
        def number_to_text_in_string(input_string: str, create_false_premise: bool = False):
            numbers = re.findall(r'-?\d+(?:\.\d+)?', input_string)
            if len(numbers) != 1: return None # original node must contain strictly 1 number

            numbers = [num for num in numbers if len(num) < 4]
            if not numbers: return None

            try:
                number = numbers[0]
                if create_false_premise:
                    adjustment = random.uniform(0.5, 4.0) * abs(float(number)) * random.choice([-1.0, 1.0])
                    adjusted_number = float(number) + adjustment
                    text = num2words(adjusted_number)
                else:
                    text = num2words.num2words(float(number))
                input_string = input_string.replace(number, text)
                return input_string
            except Exception as e:
                print(f"Error in '_knowledge_graph_paraphrase_quantity_statistics_number': {e}")
                return None
        
        SUITABLE_NODE_TYPES = ["Quantity", "Statistic", "Number"]
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["tail"])
        
        mapper = {node: number_to_text_in_string(node, create_false_premise = create_false_premise) for node in nodes_to_paraphrase}
        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if mapper.get(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        paraphrase_mapper = {k: mapper.get(k) for k in nodes_to_paraphrase}

        return paraphrase_mapper


    def knowledge_graph_paraphrase(self, 
                                   masked_kg: List[Dict[str, str]], 
                                   wikidump_date: str,
                                   create_false_premise: bool = False):
        paraphrase_type_2_func = {
            "date": self._knowledge_graph_paraphrase_date,
            "person": self._knowledge_graph_paraphrase_person,
            "year": self._knowledge_graph_paraphrase_year,
            "country": self._knowledge_graph_paraphrase_country,
            "number": self._knowledge_graph_paraphrase_quantity_statistics_number
        }

        all_paraphrases_to_choose = {}
        for ptype in paraphrase_type_2_func:
            temp = paraphrase_type_2_func[ptype](
                masked_kg = masked_kg,
                max_node = 10,
                wikidump_date = wikidump_date,
                create_false_premise = create_false_premise
            )
            if temp: all_paraphrases_to_choose[ptype] = temp

        if not all_paraphrases_to_choose: return {"masked_kg": None, "paraphrase": None}

        # choose one type of node to do paraphrasing
        _type = random.choice(list(all_paraphrases_to_choose.keys()))
        paraphrase = {}
        if _type in ["date", "year"]:
            paraphrase.update(all_paraphrases_to_choose.get("year", {}))
            paraphrase.update(all_paraphrases_to_choose.get("date", {}))
        else: paraphrase = all_paraphrases_to_choose[_type]

        res = []
        for triplet in masked_kg:
            res.append({
                "head": triplet["head"] if triplet["head"] not in paraphrase else paraphrase[triplet["head"]],
                "head_unmasked": triplet["head_unmasked"] if triplet["head_unmasked"] not in paraphrase else paraphrase[triplet["head_unmasked"]],
                "head_type": triplet["head_type"],
                "relation": triplet["relation"],
                "tail": triplet["tail"] if triplet["tail"] not in paraphrase else paraphrase[triplet["tail"]],
                "tail_unmasked": triplet["tail_unmasked"] if triplet["tail_unmasked"] not in paraphrase else paraphrase[triplet["tail_unmasked"]],
                "tail_type": triplet["tail_type"]
            })

        return {"masked_kg": res, "paraphrase": paraphrase, "false_premise": create_false_premise}
    


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
    
    def _get_bad_nodes(self, all_nodes: List[str], keypoints: Optional[List[str]] = None, knowledge_graph: Optional[List[Dict[str, str]]] = None, verbose: bool = False):
        # there are several types of nodes that we define as bad. These "bad" nodes will not be masked

        # nodes that are plural (contain "and", "&", "et al")
        PLURAL_DETECTION_KEYWORDS = ["and", "&", "et al", "et. al", "etal",
                                     " + ", " plus ", " with "]
        plural_nodes = set([node for node in all_nodes if any([kw in node for kw in PLURAL_DETECTION_KEYWORDS])])

        # nodes that overlap with other nodes, because when we mask one, it will still be there in another
        overlapping_nodes = set()
        for node1 in all_nodes:
            for node2 in all_nodes:
                if node1 != node2 and (fuzz.partial_ratio(self.text_splitter(node1), self.text_splitter(node2)) >= 75 or fuzz.partial_ratio(node1.lower(), node2.lower()) >= 75):
                    overlapping_nodes.add(node1)
                    overlapping_nodes.add(node2)

        is_entity = self.helper.is_proper_noun_phrase(all_nodes, "\n".join(keypoints))
        non_entity_nodes = {node for i, node in enumerate(all_nodes) if not is_entity[i]}

        # nodes that appear in relation
        nodes_in_relations = set([])
        for node in all_nodes:
            for triplet in knowledge_graph:
                relation = triplet["relation"]
                if fuzz.partial_ratio(self.text_splitter(node), self.text_splitter(relation)) >= 75 or fuzz.partial_ratio(node.lower(), relation.lower()) >= 75:
                    nodes_in_relations.add(node)


        # nodes that do not cover all keypoints
        try:
            lack_coverage_nodes = {triplet["head"] for triplet in knowledge_graph if len(triplet["head_coverage"]) != len(keypoints)}
        except KeyError as e:
            lack_coverage_nodes = set()

        bad_nodes = overlapping_nodes | non_entity_nodes | lack_coverage_nodes | nodes_in_relations | plural_nodes

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

        if verbose:
            print("Overlapping:", overlapping_nodes)
            print("No entity:", non_entity_nodes)
            print("Lack coverage:", lack_coverage_nodes)
            print("In relation:", nodes_in_relations)
            print("Plural:", plural_nodes)
            print("Not found in kp:", nodes_not_found_in_keypoints)
            print("Non exclusive:", nonexclusive_nodes)

        # # nodes that are connected to more than 2 other nodes
        # graph, _ = self._convert_kg_to_nx_graph(knowledge_graph)
        # dense_nodes = set([node for node in all_nodes if graph.degree[node] > 2])

        # bad_nodes = bad_nodes | dense_nodes

        return bad_nodes
    
    def _mst_from_root(self, masked_kg: List[Dict[str, str]], directed: bool = False):
        graph, _ = self._convert_kg_to_nx_graph(masked_kg)
        
        start_node = "<Unknown> #1"
        
        if start_node not in graph:
            return []
            
        undirected_for_cc = graph.to_undirected()
        cc_nodes = nx.node_connected_component(undirected_for_cc, start_node)
        
        graph = graph.subgraph(cc_nodes).copy()

        if not directed:
            graph = graph.to_undirected()
            mst = nx.Graph()
        else:
            mst = nx.DiGraph() 

        visited = set()
        min_heap = [(0, None, start_node)]
        total_cost = 0

        while min_heap:
            cost, u, v = heapq.heappop(min_heap)

            if v in visited:
                continue

            visited.add(v)
            mst.add_node(v)
            
            if u is not None:
                mst.add_edge(u, v, weight=cost)
                total_cost += cost

            for neighbor, edge_data in graph[v].items():
                if neighbor not in visited:
                    edge_weight = edge_data.get('weight', 1) 
                    heapq.heappush(min_heap, (edge_weight, v, neighbor))

        result = []
        for triplet in masked_kg:
            head, tail = triplet["head"], triplet["tail"]
            
            if directed:
                if mst.has_edge(head, tail):
                    result.append(triplet)
            else:
                if mst.has_edge(head, tail) or mst.has_edge(tail, head):
                    result.append(triplet)
                    
        return result



    def _prune_masked_graph(self, masked_kg: List[Dict[str, str]], multi_hop = True):
        if multi_hop:
            masked_subset_graph, _ = self._convert_kg_to_nx_graph([rel for rel in masked_kg if "<Unknown>" in rel["head"] and "<Unknown>" in rel["tail"]])

            masked_nodes = masked_subset_graph.nodes()
            non_leaf_masked_nodes = [node for node in masked_subset_graph.nodes() if len(masked_subset_graph[node])]
        else:
            graph, _ = self._convert_kg_to_nx_graph(masked_kg)

            masked_nodes = [node for node in graph.nodes() if "<Unknown>" in node]
            non_leaf_masked_nodes = []

        pruned_graph = []
        for triplet in masked_kg:
            head = triplet["head"]
            tail = triplet["tail"]
            if (head in non_leaf_masked_nodes and tail not in masked_nodes) \
                or (tail in non_leaf_masked_nodes and head not in masked_nodes)\
                or (head not in masked_nodes and tail not in masked_nodes):
                continue
            pruned_graph.append(triplet)

        return pruned_graph

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
            if knowledge_graph[i].get("head") \
                and knowledge_graph[i].get("head") not in bad_nodes \
                and graph.successors(knowledge_graph[i].get("head")) <= 3:
                traversal_order = list(range(i, len(knowledge_graph))) + list(range(0, i))
                break
        
        if traversal_order is None: return []
        first_edge = knowledge_graph[traversal_order[0]]

        masked_entities_info = [[first_edge["head"], first_edge["head_type"]]]
        masked_entities_names = [first_edge["head"]]


        for j in range(1, len(traversal_order)):
            i = traversal_order[j]
            relation = knowledge_graph[i]
            # is_jump = all([knowledge_graph[j-k]["tail"] != relation["head"] for k in range(1, j + 1)])

            if all([
                len(masked_entities_names) < max_nodes_to_mask, # number of masked nodes has not exceed limit
                # not is_jump, # current head is the tail of the previous relation
                any([graph.has_edge(masked_node, relation["head"]) for masked_node in masked_entities_names]), # must be the tail of some previously masked node in some relation
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
    
    def knowledge_graph_masking_single_hop(self, 
                                            knowledge_graph: List[Dict[str, str]], 
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
        
        if traversal_order is None: return None
        first_edge = knowledge_graph[traversal_order[0]]

        masked_entities_info = [[first_edge["head"], first_edge["head_type"]]]
        masked_entities_names = [first_edge["head"]]

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
            masked_kg = self._mst_from_root(masked_kg, directed = True)

        masked_keypoints_str = keypoints_str[:]
        for ent_name, ent_type in masked_entities_info:
            masked_keypoints_str = masked_keypoints_str.replace(ent_name, f"<Unknown #{masked_entities_names.index(ent_name) + 1} ({ent_type})>")

        return {"masked_kg": masked_kg, "num_hops": 1, "masked_keypoints_str": masked_keypoints_str, "keypoints_str": keypoints_str}


    def knowledge_graph_masking_single_hop_v2(self, 
                                            knowledge_graph: List[Dict[str, str]], 
                                            keypoints: List[str]):
        
        graph, heads = self._convert_kg_to_nx_graph(knowledge_graph)

        bad_nodes = self._get_bad_nodes(list(graph.nodes()), keypoints, knowledge_graph)

        if keypoints: keypoints_str = "\n".join(keypoints)
        else: keypoints_str = ""

        previously_masked_nodes = set([])

        for start in range(len(knowledge_graph)):
            # check if this traversal order is good or not
            traversal_order_check = knowledge_graph[start].get("head") \
                and knowledge_graph[start].get("head") not in bad_nodes \
                and knowledge_graph[start].get("head") not in previously_masked_nodes
            
            if not traversal_order_check: continue

            traversal_order = list(range(start, len(knowledge_graph))) + list(range(0, start))
            first_edge = knowledge_graph[traversal_order[0]]

            masked_entities_info = [[first_edge["head"], first_edge["head_type"]]]
            masked_entities_names = [first_edge["head"]]
            previously_masked_nodes.add(first_edge["head"])

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
            masked_kg = self._mst_from_root(masked_kg, directed = True)

            masked_keypoints_str = keypoints_str[:]
            for ent_name, ent_type in masked_entities_info:
                masked_keypoints_str = masked_keypoints_str.replace(ent_name, f"<Unknown #{masked_entities_names.index(ent_name) + 1} ({ent_type})>")

            yield {"masked_kg": masked_kg, "num_hops": 1, "masked_keypoints_str": masked_keypoints_str, "keypoints_str": keypoints_str}

        

    def knowledge_graph_masking_multi_hop(self,
                                          masked_kg_1, masked_kg_2):

        num_hop_1 = masked_kg_1["num_hops"]
        num_hop_2 = masked_kg_2["num_hops"]
        keypoints_str_1 = masked_kg_1["keypoints_str"]
        keypoints_str_2 = masked_kg_2["keypoints_str"]

        assert num_hop_2 == 1 and num_hop_1 == 1, "We have not implemented to work for other cases"
        # checks
        # the first check: masked_kg_2 should have num_hop = 1; the first non-masked tail of masked_kg_1 should be the masked head of masked_kg_2
        temp = {(triplet["tail"], triplet["tail_type"]): i for i, triplet in enumerate(masked_kg_1["masked_kg"]) if triplet["head"] == "<Unknown> #1"}
        check = temp.get((masked_kg_2["masked_kg"][0]["head_unmasked"], masked_kg_2["masked_kg"][0]["head_type"]), None)

        if check is None: return None
        

        combined_knowledge_graph = [
            {"head": triplet["head_unmasked"], 
             "head_type": triplet["head_type"], 
             "relation": triplet["relation"],
             "tail": triplet["tail_unmasked"], 
             "tail_type": triplet["tail_type"]}
        for triplet in masked_kg_1["masked_kg"] + masked_kg_2["masked_kg"]]

        graph, heads = self._convert_kg_to_nx_graph(combined_knowledge_graph)

        bad_nodes = self._get_bad_nodes(list(graph.nodes()), [keypoints_str_1, keypoints_str_2], combined_knowledge_graph)

        masked_node_1 = masked_kg_1["masked_kg"][0]["head_unmasked"]
        masked_node_2 = masked_kg_2["masked_kg"][0]["head_unmasked"]
        if masked_node_1 in bad_nodes or masked_node_2 in bad_nodes: return None

        masked_node_1_type = masked_kg_1["masked_kg"][0]["head_type"]
        masked_node_2_type = masked_kg_2["masked_kg"][0]["head_type"]
        if fuzz.partial_ratio(self.text_splitter(masked_node_1_type), self.text_splitter(masked_node_2_type)) >= 75 or \
            fuzz.partial_ratio(masked_node_1_type.lower(), masked_node_2_type.lower()) >= 75: return None

        masked_node_type_1 = masked_kg_1["masked_kg"][0]["head_type"]
        masked_node_type_2 = masked_kg_2["masked_kg"][0]["head_type"]

        masked_entities_names = [masked_node_1, masked_node_2]
        masked_entities_info = [
            [masked_node_1, masked_node_type_1],
            [masked_node_2, masked_node_type_2],
        ]

        combined_masked_kg = []
        for i in range(len(combined_knowledge_graph)):
            relation = combined_knowledge_graph[i]
            combined_masked_kg.append({
                "head": relation["head"] if relation["head"] not in masked_entities_names else f"<Unknown> #{masked_entities_names.index(relation['head']) + 1}",
                "head_unmasked": relation["head"],
                "head_type": relation["head_type"],
                "relation": relation["relation"],
                "tail": relation["tail"] if relation["tail"] not in masked_entities_names else f"<Unknown> #{masked_entities_names.index(relation['tail']) + 1}",
                "tail_unmasked": relation["tail"],
                "tail_type": relation["tail_type"]
            })
        masked_keypoints_str = keypoints_str_1 + "\n" + keypoints_str_2
        for ent_name, ent_type in masked_entities_info:
            masked_keypoints_str = masked_keypoints_str.replace(ent_name, f"<Unknown #{masked_entities_names.index(ent_name) + 1} ({ent_type})>")

        return {"masked_kg": combined_masked_kg, "num_hops": num_hop_1 + num_hop_2, "masked_keypoints_str": masked_keypoints_str, "keypoints_str": keypoints_str_1 + "\n" + keypoints_str_2}

    def masked_knowledge_graph_combining(self, masked_kg_1, masked_kg_2):

        if not masked_kg_1.get("masked_kg") or not masked_kg_2.get("masked_kg"):
            return None

        num_hop_1 = masked_kg_1["num_hops"]
        num_hop_2 = masked_kg_2["num_hops"]
        keypoints_str_1 = masked_kg_1["keypoints_str"]
        keypoints_str_2 = masked_kg_2["keypoints_str"]

        assert num_hop_2 == 1, "masked_kg_2 must remain a single-hop graph."

        # checks
        temp = set([(triplet["tail"], triplet["tail_type"]) 
                    for triplet in masked_kg_1["masked_kg"] 
                    if triplet["head"].startswith("<Unknown>") and not triplet["tail"].startswith("<Unknown>")])

        check = masked_kg_2["masked_kg"][0]["head"].startswith("<Unknown>") \
            and (masked_kg_2["masked_kg"][0]["head_unmasked"], masked_kg_2["masked_kg"][0]["head_type"]) in temp

        if not check: return None

        combined_knowledge_graph = [
            {"head": triplet["head_unmasked"], 
             "head_type": triplet["head_type"], 
             "relation": triplet["relation"],
             "tail": triplet["tail_unmasked"], 
             "tail_type": triplet["tail_type"]}
        for triplet in masked_kg_1["masked_kg"] + masked_kg_2["masked_kg"]]

        graph, heads = self._convert_kg_to_nx_graph(combined_knowledge_graph)
        bad_nodes = self._get_bad_nodes(list(graph.nodes()), [keypoints_str_1, keypoints_str_2], combined_knowledge_graph)

        masked_entities_names = []
        masked_entities_types = []
        
        for triplet in masked_kg_1["masked_kg"]:
            if triplet["head"].startswith("<Unknown>") and triplet["head_unmasked"] not in masked_entities_names:
                masked_entities_names.append(triplet["head_unmasked"])
                masked_entities_types.append(triplet["head_type"])
                
        # Append the node to mask from kg_2
        masked_entities_names.append(masked_kg_2["masked_kg"][0]["head_unmasked"])
        masked_entities_types.append(masked_kg_2["masked_kg"][0]["head_type"])

        # Check for bad nodes among our specifically masked entities
        if any([item in bad_nodes for item in masked_entities_names]): return None
        
        # Check for type similarities
        for i in range(len(masked_entities_types)):
            for j in range(i + 1, len(masked_entities_types)):
                if fuzz.partial_ratio(self.text_splitter(masked_entities_types[i]), 
                                      self.text_splitter(masked_entities_types[j])) >= 75 or \
                    fuzz.partial_ratio(masked_entities_types[i].lower(), 
                                      masked_entities_types[j].lower()) >= 75: return None

        # Re-mask the combined graph
        combined_masked_kg = []
        for relation in combined_knowledge_graph:
            combined_masked_kg.append({
                "head": f"<Unknown> #{masked_entities_names.index(relation['head']) + 1}" if relation["head"] in masked_entities_names else relation["head"],
                "head_unmasked": relation["head"],
                "head_type": relation["head_type"],
                "relation": relation["relation"],
                "tail": f"<Unknown> #{masked_entities_names.index(relation['tail']) + 1}" if relation["tail"] in masked_entities_names else relation["tail"],
                "tail_unmasked": relation["tail"],
                "tail_type": relation["tail_type"]
            })
        combined_masked_kg = self._mst_from_root(combined_masked_kg, directed = True)

        # Update the Keypoints strings
        masked_keypoints_str = keypoints_str_1 + "\n" + keypoints_str_2
        for i, (ent_name, ent_type) in enumerate(zip(masked_entities_names, masked_entities_types)):
            masked_keypoints_str = masked_keypoints_str.replace(ent_name, f"<Unknown #{i + 1} ({ent_type})>")

        return {
            "masked_kg": combined_masked_kg, 
            "num_hops": num_hop_1 + num_hop_2, 
            "masked_keypoints_str": masked_keypoints_str, 
            "keypoints_str": keypoints_str_1 + "\n" + keypoints_str_2
        }
    

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
        def format_user_prompt(generated_questions: List[str], relation_texts: List[str], question_target_type: str, masked_keypoints_str: Optional[str]):
            assert len(generated_questions) == len(relation_texts)
            
            res = []
            for relation_text, generated_question in zip(relation_texts, generated_questions[1:] + [""]):
                to_append = f"Relation: {relation_text}\nGenerated question: {generated_question}"
                res.append(to_append)

            res = "\n\n".join(res)
            user_prompt = f"""For this input, you will be provided context for additional information. 
User input:
Context: {masked_keypoints_str}

The generated question must ask about an a/an '{question_target_type}'
Question generation steps:\n{res}"""
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

            user_prompt = format_user_prompt(generated_questions, relation_texts, question_target_type, masked_keypoints_str)

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
    

    def question_refinement(self, generated_question: str, masked_keypoints_str: str, masked_kg: List[Dict[str, str]], wikidump_date: str):
        # wikidump_date is considered the date when the question is asked

        question_target_type = masked_kg[0]["head_type"]

        user_prompt = self.question_refinement_prompt["user"][:]\
        .replace("[ADD_KEYPOINTS_HERE]", masked_keypoints_str)\
        .replace("[ADD_QUESTION_HERE]", generated_question)\
        .replace("[ADD_QUESTION_TARGET_TYPE]", question_target_type)\
        .replace("[ADD_QUESTION_DATE]", wikidump_date)

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
                 graph_completeness_check_prompt = None,
                 LLM = None):
        
        if minicheck_cache_dir and minicheck_model_name:
            from minicheck.minicheck import MiniCheck
            self.minicheck_scorer = MiniCheck(model_name=minicheck_model_name, cache_dir=minicheck_cache_dir)
        else: self.minicheck_scorer = None
        
        self.stemmer = nltk.stem.PorterStemmer()

        self.LLM = LLM
        self.question_answerability_check_prompt = question_answerability_check_prompt
        self.graph_completeness_check_prompt = graph_completeness_check_prompt

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


    def check_completeness_of_extracted_kg_v2(self, knowledge_graph: List[Dict[str, str]], keypoints: List[str], word_check_threshold: float = 0.75, verbose: bool = False) -> bool:
        # first check word-coverage
        keypoints_str = "\n".join(keypoints)

        relations = [f"{rel['head']} {rel['relation']} {rel['tail']}" for rel in knowledge_graph]

        word_based_completeness = self._check_word_based_completeness(base_doc = keypoints_str, relations = relations, threshold = word_check_threshold)
        if not word_based_completeness: 
            if verbose:
                print("Failed word-based completeness check:", word_based_completeness)
            return False
        
        # if passed then we next do semantic check

        # relation statements and node type statements
        statements = list(set([f"{rel['head']} {rel['relation']} {rel['tail']}" for rel in knowledge_graph] \
            + [f"'{rel['head']}' is a/an {rel['head_type'].lower()}" for rel in knowledge_graph] \
            + [f"'{rel['tail']}' is a/an {rel['tail_type'].lower()}" for rel in knowledge_graph]))
        
        for i, stm in enumerate(statements):
            user_prompt = self.graph_completeness_check_prompt["user"][:]\
                .replace("[ADD_STATEMENT_HERE]", stm)\
                .replace("[ADD_KEYPOINTS_HERE]", keypoints_str)
        

            try:
                res = self.LLM.generate(
                    system_prompt = self.graph_completeness_check_prompt["system"],
                    user_prompt = user_prompt,
                    max_output_tokens = 128,
                ).strip()
            except Exception as e:
                print(e)
                return False
            
            if "A." in res: return False
        
        return True

    def kg_groundedness_filtering(
        self, 
        knowledge_graph: List[Dict[str, str]], 
        url_contents: List[Dict], 
        chunk_size: int = 5000,
        verbose: bool = False,
        num_chunks_considered: int = 5,
    ) -> List[Dict[str, str]]:
        
        if not url_contents: return []

        all_chunks = []
        all_published_dates = []
        all_urls = []
        for line in url_contents:
            content = line.get("url_content")
            published_date = line.get("published_date")
            url = line.get("url")

            if not content or not published_date or not url: continue

            if len(content) > chunk_size:
                _chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)][:num_chunks_considered]
                all_chunks.extend(_chunks)
                all_published_dates.extend([published_date] * len(_chunks))
                all_urls.extend([url] * len(_chunks))
            else:
                all_chunks.append(content)
                all_published_dates.append(published_date)
                all_urls.append(url)

        def _check_single_statement(statement, chunks, urls, published_dates):
            system_prompt = (
                "You are a strict fact-checker. Verify if the Statement is supported by the Context. "
                "Respond ONLY with 'YES' or 'NO'. A may be is likely a 'NO'"
            )

            assert len(chunks) == len(urls) == len(published_dates)
            
            for chunk, url, published_date in zip(chunks, urls, published_dates):
                user_prompt = f"URL: {url}\nPublished date: {published_date}\n\nContext: {chunk}\n\nStatement: {statement}\n\nIs this true? (YES/NO)"
                try:
                    resp = self.LLM.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_output_tokens=16
                    ).strip().upper()
                    
                    if resp.startswith("YES"):
                        return True
                except Exception as e:
                    if verbose: print(f"Error checking statement: {e}")
                    continue
            return False

        def _evaluate_relation(rel: Dict[str, str]) -> Dict[str, str]:
            rel_stmt = f"[{rel['head']} ({rel['head_type']})] {rel['relation']} [{rel['tail']} ({rel['tail_type']})]"
            
            if _check_single_statement(rel_stmt, all_chunks, all_urls, all_published_dates):
                return rel
            return None

        final_kg = []
        
        MAX_THREADS = 8
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(_evaluate_relation, rel) for rel in knowledge_graph]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        final_kg.append(result)
                except Exception as e:
                    if verbose: print(f"Thread execution error: {e}")

        return final_kg

    def kg_groundedness_filtering_batched(
        self, 
        knowledge_graph: List[Dict[str, str]], 
        url_contents: List[Dict], 
        chunk_size: int = 5000,
        verbose: bool = False,
        num_chunks_considered: int = 5,
    ) -> List[Dict[str, str]]:
        
        if not url_contents or not knowledge_graph: return []

        # 1. Extract and chunk text (Unchanged)
        all_chunks = []
        all_published_dates = []
        all_urls = []
        for line in url_contents:
            content = line.get("url_content")
            published_date = line.get("published_date")
            url = line.get("url")

            if not content or not published_date or not url: continue

            if len(content) > chunk_size:
                _chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)][:num_chunks_considered]
                all_chunks.extend(_chunks)
                all_published_dates.extend([published_date] * len(_chunks))
                all_urls.extend([url] * len(_chunks))
            else:
                all_chunks.append(content)
                all_published_dates.append(published_date)
                all_urls.append(url)

        # 2. Map relations to IDs for tracking
        pending_ids = set()
        statements_map = {}
        for i, rel in enumerate(knowledge_graph):
            stmt = f"[{rel.get('head', '')} ({rel.get('head_type', '')})] {rel.get('relation', '')} [{rel.get('tail', '')} ({rel.get('tail_type', '')})]"
            statements_map[i] = stmt
            pending_ids.add(i)
        
        supported_ids = set()

        # Update prompt to ask for a JSON list of IDs
        system_prompt = (
            "You are a strict fact-checker. You will be provided with a Context and a list of numbered Statements. "
            "Identify which Statements are explicitly supported by the Context. "
            "Respond strictly with a JSON array containing ONLY the integer IDs of the supported statements. "
            "If none are supported, return an empty array: []"
        )

        # 3. Iterate through chunks sequentially
        for chunk, url, published_date in zip(all_chunks, all_urls, all_published_dates):
            if not pending_ids:
                if verbose: print("All relations verified early. Stopping.")
                break # Early stopping: All statements have been verified!

            # Build the list of statements that are currently still pending
            statements_text = "\n".join([f"{i}: {statements_map[i]}" for i in pending_ids])
            
            user_prompt = (
                f"URL: {url}\n"
                f"Published date: {published_date}\n\n"
                f"Context: {chunk}\n\n"
                f"Statements:\n{statements_text}\n\n"
                f"Supported Statement IDs (JSON array):"
            )
            
            try:
                # We need slightly more tokens to output a JSON array like [0, 1, 3, 4]
                resp = self.LLM.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=64 
                ).strip()
                
                # Use regex to safely extract the JSON array (handles if the LLM outputs ```json ... ```)
                match = re.search(r'\[.*?\]', resp, re.DOTALL)
                if match:
                    supported_list = json.loads(match.group(0))
                    
                    if isinstance(supported_list, list):
                        # Safely remove newly verified IDs from the pending set
                        for sid in supported_list:
                            if sid in pending_ids:
                                supported_ids.add(sid)
                                pending_ids.remove(sid)
            except Exception as e:
                if verbose: print(f"Error checking batch: {e}\nResponse was: {resp}")
                continue

        # 4. Construct the final filtered KG based on supported IDs
        final_kg = [knowledge_graph[i] for i in sorted(list(supported_ids))]
        
        return final_kg
    

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
        BAD_TOKENS = ["\n"]
        if any([tok in question for tok in BAD_TOKENS]): return False

        user_prompt = self.question_answerability_check_prompt["user"][:]\
        .replace("[ADD_QUESTION_HERE]", question)

        res = self.LLM.generate(
            system_prompt = self.question_answerability_check_prompt["system"],
            user_prompt = user_prompt,
            max_output_tokens = 16
        ).strip()
        
        if verbose: print(res)

        return "B." in res
    


class RuleBasedParaphrasing:
    def __init__(self):
        from steps.utils.paraphrasing.countries import COUNTRY_PARAPHRASE
        from steps.utils.paraphrasing.years import YEAR_PARAPHRASE
        from steps.utils.paraphrasing.locations import LOCATION_PARAPHRASE
        from steps.utils.paraphrasing.numbers import NUMBER_PARAPHRASE

        self.COUNTRY_PARAPHRASE = COUNTRY_PARAPHRASE
        self.YEAR_PARAPHRASE = YEAR_PARAPHRASE
        self.LOCATION_PARAPHRASE = LOCATION_PARAPHRASE
        self.NUMBER_PARAPHRASE = NUMBER_PARAPHRASE

    def _knowledge_graph_paraphrase_date(self, 
                                         masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         wikidump_date: str,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def validate_date(date_string):
            date_string = date_string.strip()

            allowed_formats = [
                "%Y-%m-%d",       # 2025-01-28
                "%B %d, %Y",      # January 28, 2025
                "%d %B %Y",       # 28 January 2025
                "%b %d, %Y",      # Jan 28, 2025
                "%d %b %Y",       # 28 Jan 2025
                "%Y/%m/%d",       # 2025/01/28
                "%m/%d/%Y",       # 01/28/2025 (US Standard)
                "%d/%m/%Y",       # 28/01/2025 (UK/Europe/World)
                "%d.%m.%Y",       # 28.01.2025 (German/Central Europe)
                "%Y.%m.%d",       # 2025.01.28 (Standard in some Asian countries)
                "%m-%d-%Y",       # 01-28-2025
                "%d-%m-%Y",       # 28-01-2025
            ]
            for fmt in allowed_formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    return True
                except ValueError:
                    continue
            return False

        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Date":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Date":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Date":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Date":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if validate_date(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        wikidump_dt = datetime.strptime(wikidump_date, "%Y-%m-%d")

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:

            try:
                dt = dateutil.parser.parse(node_value)
                td = wikidump_dt - dt
                
                if create_false_premise:
                    td = td + timedelta(days=random.choice(range(366, 365 * 5)))

                if td.days == 0:
                    paraphrased_node_value = "today"
                elif td.days > 0:
                    # Past dates
                    paraphrased_node_value = f"{td.days} day{'s' if td.days > 1 else ''} ago"
                else:
                    # Future dates
                    abs_days = abs(td.days)
                    paraphrased_node_value = f"{abs_days} day{'s' if abs_days > 1 else ''} from now"
                    
            except Exception: 
                continue

            paraphrase_mapper[node_value] = paraphrased_node_value

        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_year(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         wikidump_date: str,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def is_valid_year_datetime(year_str):
            try:
                datetime.strptime(year_str, '%Y')
                return True
            except ValueError:
                return False
            
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Year":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Year":
                nodes_to_paraphrase.append(triplet["tail"])

        nodes_to_paraphrase = [int(node) for node in nodes_to_paraphrase if is_valid_year_datetime(node) and int(node) in self.YEAR_PARAPHRASE]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        wikidump_year = int(datetime.strptime(wikidump_date, "%Y-%m-%d").year)

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:

            if create_false_premise:
                paraphrased_node_value = str(node_value + random.choice(range(1, 11)) * random.choice([-1, 1]))
            else:
                paraphrased_node_value = self.YEAR_PARAPHRASE[int(node_value)]


            paraphrase_mapper[str(node_value)] = paraphrased_node_value


        return paraphrase_mapper
    
    def _knowledge_graph_paraphrase_person(self, masked_kg: List[Dict[str, str]], 
                                     max_node: int,
                                     create_false_premise: bool = False,
                                     **kwargs):
        def is_valid_person_name(text):
            clean_text = text.strip()
            words = clean_text.split()
            
            if len(words) <= 1:
                return False
                
            for word in words:
                if not word[0].isupper():
                    return False
            return True
        
        def abbreviate_name(name, create_false_premise=False, pure_initials=False):
            words = name.strip().split()
            if not words:
                return ""

            if create_false_premise:
                words = words[:-1] + [names.get_last_name()] 
                return " ".join(words)
            
            if pure_initials:
                return "".join([word[0].upper() + "." for word in words])

            # Standard abbreviation (for example "A. M. Turing")
            initials = [word[0].upper() + "." for word in words[:-1]]
            return " ".join(initials + [words[-1]])
        
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Person":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Person":
                nodes_to_paraphrase.append(triplet["tail"])

        unique_nodes = list(set([node for node in nodes_to_paraphrase if is_valid_person_name(node)]))
        nodes_to_paraphrase = random.sample(unique_nodes, max_node) if len(unique_nodes) > max_node else unique_nodes

        paraphrase_mapper = {}
        
        if not nodes_to_paraphrase:
            return paraphrase_mapper

        standard_abbrev_person = random.choice(nodes_to_paraphrase)

        for node_value in nodes_to_paraphrase:
            if node_value == standard_abbrev_person:
                paraphrased_node_value = abbreviate_name(node_value, create_false_premise, pure_initials=False)
            else:
                paraphrased_node_value = abbreviate_name(node_value, create_false_premise, pure_initials=True)
                
            paraphrase_mapper[node_value] = paraphrased_node_value

        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_country(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):

        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] == "Country":
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] == "Country":
                nodes_to_paraphrase.append(triplet["tail"])
        
        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if node in self.COUNTRY_PARAPHRASE]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        paraphrase_mapper = {}
        for node_value in nodes_to_paraphrase:
            if not create_false_premise:
                paraphrased_node_value = random.choice(self.COUNTRY_PARAPHRASE[node_value])
            else:
                all_countries = list(pycountry.countries)
                for _ in range(10):
                    random_country = random.choice(all_countries).name
                    if random_country != node_value: break
                paraphrased_node_value = random_country

            paraphrase_mapper[node_value] = paraphrased_node_value

        return paraphrase_mapper
    

    def _knowledge_graph_paraphrase_quantity_statistics_number(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):
        
        def generate_float_trivia_question(target_number):
            def format_num(n):
                if isinstance(n, float) and n.is_integer():
                    return int(n)
                return round(n, 5)
            
            if not isinstance(target_number, (int, float)):
                return "Error: Please provide a valid number."

            fact_val, fact_desc_ = random.choice(list(self.NUMBER_PARAPHRASE.items()))
            fact_desc = random.choice(fact_desc_)

            if math.isclose(target_number, fact_val):
                return f"[Answer to: {fact_desc}]"

            valid_strategies = []
            percentage = (target_number / fact_val) * 100
            if percentage > 0:
                valid_strategies.append(f"{format_num(percentage)}% of [Answer to: {fact_desc}]")


            if target_number != 0:
                divisor = fact_val / target_number
                if divisor.is_integer():
                    valid_strategies.append(f"[Answer to: {fact_desc}] / {int(divisor)}")

            multiplier = target_number / fact_val
            if round(multiplier, 2) == multiplier: 
                valid_strategies.append(f"[Answer to: {fact_desc}] * {format_num(multiplier)}")

            difference = target_number - fact_val
            diff_formatted = format_num(abs(difference))
            
            if difference > 0:
                valid_strategies.append(f"[Answer to: {fact_desc}] + {diff_formatted}")
            else:
                valid_strategies.append(f"[Answer to: {fact_desc}] - {diff_formatted}")

            return random.choice(valid_strategies)
        
        def number_to_text_in_string(input_string: str, create_false_premise: bool = False):
            numbers = re.findall(r'-?\d+(?:\.\d+)?', input_string)
            if len(numbers) != 1: return None

            numbers = [num for num in numbers if len(num) < 4]
            if not numbers: return None

            try:
                number = numbers[0]
                if create_false_premise:
                    adjustment = random.uniform(0.5, 4.0) * abs(float(number)) * random.choice([-1.0, 1.0])
                    adjusted_number = float(number) + adjustment
                    text = num2words.num2words(adjusted_number)
                else:
                    text = generate_float_trivia_question(float(number))
                input_string = input_string.replace(number, "N")
                input_string = input_string + f" (Where N = {text})"
                return input_string
            except Exception as e:
                print(f"Error in '_knowledge_graph_paraphrase_quantity_statistics_number': {e}")
                return None
        
        SUITABLE_NODE_TYPES = ["Quantity", "Statistic", "Number"]
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["tail"])
        
        mapper = {node: number_to_text_in_string(node, create_false_premise = create_false_premise) for node in nodes_to_paraphrase}
        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if mapper.get(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        paraphrase_mapper = {k: mapper.get(k) for k in nodes_to_paraphrase}

        return paraphrase_mapper

    def _knowledge_graph_paraphrase_location_city(self, masked_kg: List[Dict[str, str]], 
                                         max_node: int,
                                         create_false_premise: bool = False,
                                         **kwargs):
        import pyproj
        from geopy.distance import geodesic

        def get_compass_direction(bearing):
            directions = [
                "North", "North-Northeast", "Northeast", "East-Northeast", "East",
                "East-Southeast", "Southeast", "South-Southeast", "South",
                "South-Southwest", "Southwest", "West-Southwest", "West",
                "West-Northwest", "Northwest", "North-Northwest"
            ]

            index = round(bearing / 22.5) % 16
            return directions[index]
        
        def spatial_reasoning_paraphrase(random_location, target):
            lat1, lon1, city1, type1 = random_location["lat"], random_location["long"], random_location["name"], random_location["type"]
            lat2, lon2, city2, type2 = target["lat"], target["long"], target["name"], target["type"]

            coords1 = (lat1, lon1)
            coords2 = (lat2, lon2)

            distance_miles = geodesic(coords1, coords2).miles
            
            geod = pyproj.Geod(ellps='WGS84')
            fwd_azimuth, back_azimuth, _ = geod.inv(lon1, lat1, lon2, lat2)
            
            compass_bearing = (fwd_azimuth + 360) % 360
            
            readable_direction = get_compass_direction(compass_bearing)
            
            res = f"""The {type2} that is {int(distance_miles):,} miles, {readable_direction} ({compass_bearing:.1f}°) of {type1} of {city1}"""

            return res
        
        SUITABLE_NODE_TYPES = ["Location", "City"]
        nodes_to_paraphrase = []
        for triplet in masked_kg:
            if "<Unknown" not in triplet["head"] and triplet["head_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["head"])
            
            if "<Unknown" not in triplet["tail"] and triplet["tail_type"] in SUITABLE_NODE_TYPES:
                nodes_to_paraphrase.append(triplet["tail"])

        mapper = {}
        for node in nodes_to_paraphrase:
            splitted_node = [item.strip() for item in node.split(",")]
            if node in self.LOCATION_PARAPHRASE:
                target_location = node
            elif splitted_node[0] in self.LOCATION_PARAPHRASE:
                target_location = splitted_node[0]
            else: continue

            random_location = random.choice(list(self.LOCATION_PARAPHRASE.keys()))

            if create_false_premise:
                for _ in range(100):
                    target_location = random.choice(list(self.LOCATION_PARAPHRASE.keys()))
                    if target_location != random_location: break

            random_inp = self.LOCATION_PARAPHRASE[random_location]
            target_inp = self.LOCATION_PARAPHRASE[target_location]
            mapper[node] = spatial_reasoning_paraphrase(random_inp, target_inp)

        nodes_to_paraphrase = [node for node in nodes_to_paraphrase if mapper.get(node)]
        nodes_to_paraphrase = random.sample(nodes_to_paraphrase, max_node) if len(nodes_to_paraphrase) > max_node else nodes_to_paraphrase

        paraphrase_mapper = {k: mapper.get(k) for k in nodes_to_paraphrase}

        return paraphrase_mapper

        

    def _paraphrase_single_graph(self, generated_question: str, masked_keypoints_str: str, masked_kg: List[Dict[str, str]], wikidump_date, create_false_premise = False):

        paraphrased_question = generated_question[:]
        paraphrased_masked_keypoints_str = masked_keypoints_str[:]

        paraphrase_type_2_func = {
            "date": self._knowledge_graph_paraphrase_date,
            "person": self._knowledge_graph_paraphrase_person,
            "year": self._knowledge_graph_paraphrase_year,
            "country": self._knowledge_graph_paraphrase_country,
            "number": self._knowledge_graph_paraphrase_quantity_statistics_number,
            "city": self._knowledge_graph_paraphrase_location_city
        }

        all_paraphrases_to_choose = {}
        for ptype in paraphrase_type_2_func:
            temp = paraphrase_type_2_func[ptype](
                masked_kg = masked_kg,
                max_node = 10,
                wikidump_date = wikidump_date,
                create_false_premise = create_false_premise
            )
            if temp: all_paraphrases_to_choose[ptype] = temp

        if not all_paraphrases_to_choose: return {"masked_kg": None, "paraphrase": None}

        if create_false_premise is True:
            # choose one type of node to do paraphrasing
            _type = random.choice(list(all_paraphrases_to_choose.keys()))
            paraphrase = {}
            if _type in ["date", "year"]:
                paraphrase.update(all_paraphrases_to_choose.get("year", {}))
                paraphrase.update(all_paraphrases_to_choose.get("date", {}))
            else: paraphrase = all_paraphrases_to_choose[_type]
        else:
            # choose all type
            paraphrase = {}
            for ptype in all_paraphrases_to_choose.keys():
                paraphrase.update(all_paraphrases_to_choose[ptype])

        res = []
        for triplet in masked_kg:
            res.append({
                "head": triplet["head"] if triplet["head"] not in paraphrase else paraphrase[triplet["head"]],
                "head_unmasked": triplet["head_unmasked"] if triplet["head_unmasked"] not in paraphrase else paraphrase[triplet["head_unmasked"]],
                "head_type": triplet["head_type"],
                "relation": triplet["relation"],
                "tail": triplet["tail"] if triplet["tail"] not in paraphrase else paraphrase[triplet["tail"]],
                "tail_unmasked": triplet["tail_unmasked"] if triplet["tail_unmasked"] not in paraphrase else paraphrase[triplet["tail_unmasked"]],
                "tail_type": triplet["tail_type"]
            })

        # for k, v in paraphrase.items():
        #     paraphrased_question = paraphrased_question.replace(k, v)
        #     paraphrased_masked_keypoints_str = paraphrased_masked_keypoints_str.replace(k, v)

        legend_items = []
        placeholder_idx = 1
        
        for k, v in paraphrase.items():
            placeholder = f"X_{placeholder_idx}"
            
            paraphrased_question = paraphrased_question.replace(k, placeholder)
            paraphrased_masked_keypoints_str = paraphrased_masked_keypoints_str.replace(k, placeholder)
            
            legend_items.append(f"{placeholder} is {v}")
            placeholder_idx += 1

        if legend_items:
            legend_suffix = "\nWhere " + ", ".join(legend_items)
            paraphrased_question += "\n" + legend_suffix
            paraphrased_masked_keypoints_str += "\n" + legend_suffix
        # -----------------------------

        return {"question": paraphrased_question,
                "masked_keypoints_str": paraphrased_masked_keypoints_str,
                "masked_kg": res, 
                "paraphrase": paraphrase, 
                "false_premise": create_false_premise}


    def __call__(self, inputs: List, wikidump_date: str, create_false_premise: bool):
        paraphrased_inputs = [
            {**item, **self._paraphrase_single_graph(item["question"], item["masked_keypoints_str"], item["masked_kg"], wikidump_date, create_false_premise = create_false_premise)} 
            for item in inputs
        ]

        return paraphrased_inputs
