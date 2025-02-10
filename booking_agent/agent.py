
from utils.state import State , Router , RouterState
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils.nodes import NodeUtils
from langgraph.types import Command, interrupt

builder = StateGraph(State)
# builder.add_node("call_model", call_model)
builder.add_node("get_info", NodeUtils.info_chain)
builder.add_node("ask_info_empty", NodeUtils.ask_info_empty)
builder.add_node("human", NodeUtils.human_node)
builder.add_node("human_confirm", NodeUtils.human_node_confirm)
builder.add_node("ask_confirm", NodeUtils.ask_user_confirm)

builder.add_node("cancel_book",NodeUtils.cancel_booking)

builder.add_node("confirm_change", NodeUtils.confirm_change)

builder.add_node("perform_request", NodeUtils.perform_request)

builder.add_node("human_request", NodeUtils.human_node_request)
builder.add_node("ask_request", NodeUtils.ask_request)
builder.add_node("human_ans", NodeUtils.human_ans_change)


builder.add_node("get_quotes_booking", NodeUtils.get_quotes)
builder.add_node("human_choose_quote", NodeUtils.human_choose_quote)
builder.add_node("accept_booking", NodeUtils.accept_booking) 




builder.add_edge(START, "get_info")
# builder.add_edge("get_info", "ask_confirm")
builder.add_conditional_edges("get_info", NodeUtils.get_state ,["ask_confirm", "ask_info_empty"])
builder.add_edge("ask_info_empty", "human")
builder.add_edge("ask_confirm", "human_confirm")
builder.add_conditional_edges("human_confirm", NodeUtils.get_confirm_state,["get_quotes_booking","cancel_book", "confirm_change","perform_request"])

builder.add_edge("confirm_change", "human_request")
builder.add_conditional_edges("human_request", NodeUtils.get_change_state,["ask_request", "cancel_book","perform_request"])
builder.add_edge("get_quotes_booking", "human_choose_quote")
builder.add_edge("human_choose_quote", "accept_booking")

builder.add_edge("ask_request","human_ans")
builder.add_edge("human_ans", "perform_request")
builder.add_edge("accept_booking", END)
# builder.add_edge("call_model", "info")
# checkpointer = MemorySaver()
# graph = builder.compile(checkpointer=checkpointer)
subgraph = builder.compile()


parent_graph = StateGraph(RouterState)
parent_graph.add_node(NodeUtils.router_node)
parent_graph.add_node("call_model", NodeUtils.call_model)
parent_graph.add_node("booking_graph", subgraph)
parent_graph.add_edge(START, "router_node")
parent_graph.add_conditional_edges("router_node", NodeUtils.route_after_prediction)
# parent_graph.add_edge("call_model", END)
# parent_graph.add_edge("booking_graph", END)
memory = MemorySaver()
graph = parent_graph.compile(checkpointer=memory)


import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user = input("User: ")
    if user.lower() == "q":
        print("Đã thoát chatbot.")
        break
    human_command = {"messages": user} 
    while True:  
        last_output = None  
        for output in graph.stream(human_command, config=config, stream_mode="updates", subgraphs=True):
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
                for key, value in output[1].items():
                    if isinstance(value, dict) and "messages" in value:
                        for message in value["messages"]:
                            # print(type(message))
                            if isinstance(message, tuple) and message[0] == "content":
                                print("Assitant : " + message[1])
                            if isinstance(message, dict) and "content" in message:
                                if message["role"] == "ai":
                                    print("Assitant : " + message["content"])
        
            last_output = output  
        if isinstance(last_output, tuple) and isinstance(last_output[1], dict) and "__interrupt__" in last_output[1]:
            user = input("User: ")  
            human_command = Command(resume=user)
        else:
            break 