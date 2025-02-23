from utils.state import State , Router , RouterState
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils.nodes import NodeUtils
from utils.state import BookingCarDetails
from langgraph.types import Command, interrupt
from flask import Flask , request , jsonify
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from collections import OrderedDict
import utils.state
app = Flask(__name__)
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
builder.add_node("human_ans", NodeUtils.human_ans_request)
builder.add_node("ask_change", NodeUtils.system_ask_change)

builder.add_node("human_ans_change", NodeUtils.human_node_ans_change)


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
builder.add_edge("ask_change","human_ans_change")
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



# Create a class to store user state
class UserState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.last_output = None
        self.pickup_result = None
        self.destination_result = None
        self.booking_details = BookingCarDetails(name="", number_phone="", pick_up_location="", destination_location="", pick_up_time="", flight_code="",flight="")

# Replace dictionary with list to store user states
user_states = []

def get_user_state(user_id):
    for state in user_states:
        if state.user_id == user_id:
            return state
    # Create new state if not found
    new_state = UserState(user_id)
    user_states.append(new_state)
    return new_state

def process_chat(user_input, user_id):
    # Get or create user state
    user_state = get_user_state(user_id)
    responses = []
    config = {"configurable": {"thread_id": user_id}}
    
    if user_state.last_output is None:
        human_command = {"messages": user_input}
    else:
        human_command = Command(resume=user_input)

    for output in graph.stream(human_command, config=config, stream_mode="updates", subgraphs=True):
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
            for key, value in output[1].items():
                if isinstance(value, dict) and "messages" in value:
                    for message in value["messages"]:
                        if isinstance(message, tuple) and message[0] == "content":
                            responses.append(message[1])
                        if isinstance(message, dict) and "content" in message:
                            if message["role"] == "ai":
                                responses.append(message["content"])
        
        # Update user state last_output
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
            user_state.last_output = None if "__interrupt__" not in output[1] else output

    branch_state = graph.get_state(config, subgraphs=True)
    if branch_state.tasks and branch_state.tasks[0].state:
        booking_info = branch_state.tasks[0].state.values.get('booking_info', {})
        if booking_info:
            user_state.booking_details = booking_info
            # Update pickup and destination results
            user_state.pickup_result = utils.state.pick_up_result
            user_state.destination_result = utils.state.destination_result
    return responses

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("userId", "")
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    utils.state.pick_up_result = None
    utils.state.destination_result = None    
    responses = process_chat(user_input, user_id)
    user_state = get_user_state(user_id)
    
    response_data = OrderedDict([
        ("context", responses),
        ("booking_details", user_state.booking_details.model_dump()),
        ("pickup_result", user_state.pickup_result),
        ("destination_result", user_state.destination_result)
    ])
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)