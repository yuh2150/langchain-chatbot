import os
from flask import Flask, request, jsonify
from typing import Annotated, Literal , List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field , field_validator, ValidationInfo
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.graph.message import AnyMessage
from API.booking import BookingAPI
from API.geoCoding import GeoCodingAPI
from API.getKey import OAuthClient
from API.getQuotes import QuotesAPI
from API.is_Airport import IsAirport

from langgraph.graph import StateGraph, MessagesState, START, END

from quotes import Quote

llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
jupiterAPI = os.getenv('JUPITER_API')
quoteAPI = str(jupiterAPI) + "/demand/v1/quotes"
bookingsAPI  = str(jupiterAPI) + '/demand/v1/bookings'

class BookingCarDetails(BaseModel):
    """Details for the bookings car details"""
    name: str = Field(
        ...,
        description="The name of the person booking the ride. Do not autofill if not provided",
    )
    number_phone: str = Field(
        ...,
        description="The phone number of the user. Do not autofill if not provided",
    )
    pick_up_location: str = Field(
        ...,
        description="The location where the user will be picked up. This can be a full address or a specific location name. Do not autofill if not provided",
    )
    destination_location: str = Field(
        ...,
        description="The destination location for the ride. This can be a full address or a specific location name. Do not autofill if not provided"
    )
    pick_up_time: str = Field(
        ...,
        description="The time the user intends to be picked up. No format keeps the text related to time. Do not autofill if not provided"
    )
    @field_validator('pick_up_location')
    @classmethod
    def validate_pickup(cls, value:str):
        geoCodingAPI = GeoCodingAPI()
        if value == '':
            return ''
        else :
            geoCoding_pickup = geoCodingAPI.get_geocoding(value)
            if geoCoding_pickup["status"] == "OK" :
                return geoCoding_pickup['results'][0]['formatted_address']
            else:
                raise ValueError(f"Invalid pick-up location: {value}")
    @field_validator('destination_location')
    @classmethod
    def validate_destination(cls, value : str, info: ValidationInfo):
        geoCodingAPI = GeoCodingAPI()
        
        # print (geoCoding_destination['results'][0]['formatted_address'])
        if value == '':
            return ''
        else :
            geoCoding_destination = geoCodingAPI.get_geocoding(value)
            if geoCoding_destination["status"] == "OK":
                if geoCoding_destination['results'][0]['formatted_address'] == info.data['pick_up_location']:
                    raise ValueError(f"Invalid destination location: {value}")
                else:
                    return geoCoding_destination['results'][0]['formatted_address']
            else:
            
                raise ValueError(f"Invalid destination location: {value}")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    quote_id: str
    booking_info: BookingCarDetails 
    slot_empty: list
def check_what_is_empty(user_personal_details):
    ask_for = []
    for field, value in user_personal_details.model_dump().items():
        if value in [None, "", 0]:  
            ask_for.append(field)
    return ask_for


def add_non_empty_details(current_details: BookingCarDetails, new_details: BookingCarDetails):
    non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    updated_details = current_details.model_copy(update=non_empty_details)
    return updated_details
def update_details(current_details: BookingCarDetails, new_details: BookingCarDetails , field : str):
    # non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    updated_details = current_details.model_copy(update=field)
    return updated_details

def ask_for_info(ask_list: list):
    first_prompt = ChatPromptTemplate.from_template(
        """Ask one question at a time, even if you don't get all the info. Don't list the questions or greet the user. 
        ### ask_for list: {ask_for}"""
    )

    info_gathering_chain = first_prompt | llm | StrOutputParser()
    ai_chat = info_gathering_chain.invoke({"ask_for": ask_list})
    return ai_chat
def filter_response(text_input, user_details : BookingCarDetails ):
    chain = llm.with_structured_output(BookingCarDetails)
    res = chain.invoke(text_input)
    # add filtered info to the
    user_details = add_non_empty_details(user_details,res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for
def ask_confirm_info(booking_details: BookingCarDetails):
    # booking_details.
    message = (
        f"Please confirm your ride details:\n"
        f"- Pickup Location: {booking_details.pick_up_location}\n"
        f"- Destination: {booking_details.destination_location}\n"
        f"- Pickup Time: {booking_details.pick_up_time}\n"
        f"- Name: {booking_details.name}\n"
        f"- Contact Number: {booking_details.number_phone}\n"
    )
    print(message)
def call_model (state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
@tool
def get_booking_details(state : MessagesState):
    """ Call function to get the details for a booking from user"""
    # state = State(booking_info=BookingCarDetails(name="", number_phone="", pick_up_location="", destination_location="", pick_up_time=""))
    chain = llm.with_structured_output(BookingCarDetails)
    response = chain.invoke(state["messages"][-1].content)
    booking_details = BookingCarDetails(name="", number_phone="", pick_up_location="", destination_location="", pick_up_time="")
    user_details = add_non_empty_details(booking_details, response)
    ask_for = check_what_is_empty(user_details)
    while ask_for:  
        ai_response = ask_for_info(ask_for)
        print(ai_response)
        # print(ai_response)
        text_input = interrupt(ai_response)
        print("hi" + text_input)
        user_details, ask_for = filter_response(text_input, user_details)
        return {"human_input": text_input}
    
    return user_details
# @tool
def get_quotes(booking_details : BookingCarDetails):
    """Call function to fetches quotes for car bookings based on the provided booking details."""
    quotesAPI = QuotesAPI(os.getenv("JUPITER_API") + "/demand/v1/quotes")
    geoCodingAPI = GeoCodingAPI()
    # geoCoding_destination =
    geoCoding_pickup =  geoCodingAPI.get_geocoding(booking_details.pick_up_location)
    geoCoding_destination = geoCodingAPI.get_geocoding(booking_details.destination_location)
    # input_datetime = datetime.fromisoformat(pick_up_time)
    pickup_datetime = "2025-01-23T09:24:10.000Z"
    
    pickup_coords = { "latitude": float(geoCoding_pickup['results'][0]['geometry']['location']['lat']),"longitude": float(geoCoding_pickup['results'][0]['geometry']['location']['lng']),}
    destination_coords = { "latitude": float(geoCoding_destination['results'][0]['geometry']['location']['lat']),"longitude": float(geoCoding_destination['results'][0]['geometry']['location']['lng']),}
    quotes_data = quotesAPI.get_quotes(pickup_datetime, pickup_coords, destination_coords)
    quotes = []
    for item in quotes_data:
        quote = Quote(
        quote_id=item['quoteId'],
        expires_at=item['expiresAt'],
        vehicle_type=item['vehicleType'],
        price_value=item['price']['value'],
        price_currency=item['price']['currency'] if 'currency' in item['price'] and item['price']['currency'] is not None else 'CAD',
        luggage=item['luggage'],
        passengers=item['passengers'],
        provider_name=item['provider']['name'],
        provider_phone=item['provider']['phone']
        )
        quotes.append(quote)

    for quote in quotes:
        print({
            "title": f"{quote.vehicle_type} - {quote.price_value} {quote.price_currency}",
            "payload": f"{quote.quote_id}"
        })
    print(quotes)
    return "15$"
@tool
def change_info(fields : List[str], booking_details : BookingCarDetails):
    """Change the booking details field """
    for field in fields:
        ai_response = ask_for_info([field])
        print(ai_response)
        text_input = input()
        chain = llm.with_structured_output(BookingCarDetails)
        response =chain.invoke(text_input)
        user_details = add_non_empty_details(booking_details,response)
    return user_details 
@tool
def accept_booking(quote_Id: str ,booking_details : BookingCarDetails ):
    """Call function to accept booking with quote_ID."""
    bookingAPI = BookingAPI(bookingsAPI)
    # quote_id = tracker.get_slot("quoteId")
    person_name = booking_details.name
    number_contact = booking_details.number_phone
    
    passenger_info = {
        "title": "Mr",
        "phone": number_contact,
        "firstName": person_name,
        "lastName": ""
    }

    response = bookingAPI.create_booking(
        quote_id=quote_Id,
        passenger_info=passenger_info
    )
    return response
def call_model(state : State ):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
def info_chain(state : State):

    if "booking_info" in state:
        booking_details = state["booking_info"]
    else:
        booking_details = BookingCarDetails(name="", number_phone="", pick_up_location="", destination_location="", pick_up_time="")
    
    chain = llm.with_structured_output(BookingCarDetails)
    response = chain.invoke(state["messages"][-1].content)

    user_details = add_non_empty_details(booking_details, response)
    ask_for = check_what_is_empty(user_details)
    # messages = get_messages_info(state["messages"])
    # response = llm_with_tool.invoke(messages)
    return Command(update={"slot_empty": ask_for , "booking_info": user_details})
def human_node(
    state: State, config
)-> Command[Literal["get_info"]] :
    """A node for collecting user input."""
    
    user_input = interrupt(value="Ready for user input.")

    # # identify the last active agent
    # # (the last active node before returning to human)
    # langgraph_triggers = config["metadata"]["langgraph_triggers"]
    # if len(langgraph_triggers) != 1:
    #     raise AssertionError("Expected exactly 1 trigger in human node")

    # active_agent = langgraph_triggers[0].split(":")[1]
    # print("Active agent")
    # print(active_agent)
    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        # goto=active_agent,
    )
def ask_info_empty(
    state: State,
):
    first_prompt = ChatPromptTemplate.from_template(
        """Ask one question at a time, even if you don't get all the info. Don't list the questions or greet the user. 
        ### ask_for list: {ask_for}"""
    )
    info_gathering_chain = first_prompt | llm | StrOutputParser()
    if state["slot_empty"] : 
        ai_chat = info_gathering_chain.invoke({"ask_for": state["slot_empty"]})
        return Command(update= {"messages": [
                    {
                        "role": "ai",
                        "content": ai_chat,
                }
            ]} , goto="human")
    else :
        return END
def get_state(state : State):
    slot_empty = state["slot_empty"]
    if slot_empty == [] :
        return "ask_confirm"
    else : 
        return "ask_info_empty"
    
def ask_confirm(state : State):
    """Ask the user again to confirm the booking details. """
    message = (
        f"Please confirm your ride details:\n"
        f"- Pickup Location: {state["booking_info"].pick_up_location}\n"
        f"- Destination: {state["booking_info"].destination_location}\n"
        f"- Pickup Time: {state["booking_info"].pick_up_time}\n"
        f"- Name: {state["booking_info"].name}\n"
        f"- Contact Number: {state["booking_info"].number_phone}\n"
    )
    return Command(update= {"messages": [
                    {
                        "role": "ai",
                        "content": message,
                }
            ]} )

class ConfirmDetails(BaseModel):
    """
    Represents the user's confirmation intent and request details.
    """
    confirm: bool = Field(
        ..., 
        description="User confirmation intent. True if the user confirms, False if not."
    )
    request: str = Field(
        ..., 
        description="User's request to change booking details. Returns 'None' if there is no request.Returns 'Cancel' if there is cancel."
    )
    # continue_booking: bool = Field(
    #     # default= 'True',
    #     description="True if the user wants to change info and continue booking, False if they want to cancel."
    # )
    
def get_confirm_state(state: State):

    # Lấy input từ người dùng qua cơ chế interrupt
    user_input = interrupt("Please confirm the booking")
    
    return Command(
        update={"messages": [
            {
                "role": "human",
                "content": user_input
            }
        ]},
        goto="get_quotes_booking"
    )
def human_node_confirm(
    state: State, config
):
    """A node for collecting user confirm."""
    
    user_input = interrupt(value="Please confirm details booking.")
    return Command(
        update={
            "messages":  user_input,
        },
        goto="get_confirm"
    )
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("get_info", info_chain)
builder.add_node("ask_info_empty", ask_info_empty)
builder.add_node("human", human_node)
# builder.add_node("human_confirm", human_node_confirm)
builder.add_node("ask_confirm", ask_confirm)
builder.add_node("get_quotes_booking", get_quotes)

builder.add_node("get_confirm", get_confirm_state)

builder.add_edge(START, "get_info")

# builder.add_edge("get_info", "ask_confirm")

builder.add_conditional_edges("get_info", get_state ,["ask_confirm", "ask_info_empty"])
builder.add_edge("ask_info_empty", "human")
# builder.add_edge("ask_confirm", "human_confirm")
builder.add_edge("ask_confirm", "get_confirm")
# builder.add_edge("human_confirm", "get_confirm")
builder.add_edge("get_confirm", "get_quotes_booking")
# builder.add_edge("call_model", "info")
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
while True:
    user = input("User (q/Q to quit): ")
    print(f"User (q/Q to quit): {user}")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream(
        {"messages": user}, config=config, stream_mode="updates"
    ):  
        print(output)
        # last_message = next(iter(output.values()))["messages"][-1]
        # last_message.pretty_print()