import json
from rich import print as rprint
from transformers import AutoTokenizer


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True
    ):
        self.searcher = searcher
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": f"Perform a search on a knowledge source. Returns top-{self.k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": "Retrieve a full document by its docid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools
    

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "published_date": cand["published_date"], "snippet": cand["snippet"]})
            else:
                results.append({"docid": cand["docid"], "published_date": cand["published_date"], "score": cand["score"],"snippet": cand["snippet"]})

        return json.dumps(results, indent=2)
    
    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)

    



def run_conversation_with_tools(
        client, initial_request: dict, 
        tool_handler, 
        max_iterations: int = 100, 
        max_search_calls: int = 5,
        verbose: bool = False):
    # code from https://github.com/ChuanMeng/text-ranking-in-deep-research/blob/main/search_agent/oss_client.py

    tool_usage = {}
    messages = initial_request["input"]

    iteration = 1
    while iteration <= max_iterations:
        print("Iteration:",iteration)
        try:
            request = initial_request.copy()
            request["input"] = messages
            response = client.responses.create(
                **request,
        )
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
                rprint(f"Request: {request}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")


        if len(response_dict['output']) >= 1 and response_dict['output'][-1]['type'] == 'reasoning':
            continue

        for i in range(len(response_dict['output'])):
            if response_dict['output'][i]['type'] == 'mcp_call':
                # TODO: upstream vLLM bug outputs mcp_call but only accepts function_call as input
                # Convert mcp_call to function_call format
                response_dict['output'][i] = {
                    'id': response_dict['output'][i]['id'],
                    'call_id': response_dict['output'][i]['id'],  # function_call uses call_id
                    'arguments': response_dict['output'][i]['arguments'],
                    'name': response_dict['output'][i]['name'],
                    'type': 'function_call',
                    'status': None
                }

            del response_dict['output'][i]["status"]
            messages.append(response_dict['output'][i])

        function_calls = [item for item in response_dict['output'] if item['type'] in ['function_call', 'mcp_call', 'custom_tool_call']]

        if not function_calls:
            if messages[-1]['type'] != 'message':
                continue
            return messages, tool_usage, "completed"

        new_messages = messages.copy()

        for tool_call in function_calls:
            try:
                if tool_call["name"] == "search_engine" and tool_usage.get("search_engine", 0) >= max_search_calls:
                    limit_msg = f"System Error: Limit of {max_search_calls} calls reached for search_engine. You must proceed with the information you have."
                    new_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": limit_msg,
                        }
                    )
                    continue

                arguments = json.loads(tool_call["arguments"])
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1

                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": result,
                    }
                )

            except Exception as e:
                error_msg = f"Error executing {tool_call['name']}: {str(e)}"
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": error_msg,
                    }
                )
        messages = new_messages
        iteration += 1

    return messages, tool_usage, "incomplete"