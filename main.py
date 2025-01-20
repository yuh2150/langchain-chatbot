import os
import requests
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field , field_validator
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

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
jupiterAPI = os.getenv('JUPITER_API')
quoteAPI = str(jupiterAPI) + "/demand/v1/quotes"
bookingsAPI  = str(jupiterAPI) + '/demand/v1/bookings'

class BookingCarDetails(BaseModel):
    """Details for the bookings car details"""
    name: str = Field(
        ...,
        description="The name of the person booking the ride.This is optional if provided",
    )
    number_phone: str = Field(
        ...,
        description="The phone number of the user.This is optional if provided",
    )
    pick_up_location: str = Field(
        ...,
        description="The location where the user will be picked up. This can be a full address or a specific location name.This is optional if provided",
    )
    destination_location: str = Field(
        ...,
        description="The destination location for the ride. This can be a full address or a specific location name.This is optional if provided"
    )
    pick_up_time: str = Field(
        ...,
        description="The time the user intends to be picked up. No format keeps the text related to time..This is optional if provided"
    )
    @field_validator('pick_up_location',mode= "before")
    @classmethod
    def validate_pickup(cls, value):
        geoCodingAPI = GeoCodingAPI()
        geoCoding_pickup = geoCodingAPI.get_geocoding(value)
        if geoCoding_pickup["status"] == "OK" :
            return geoCoding_pickup['results'][0]['formatted_address']
        else:
            raise ValueError(f"Invalid pick-up location: {value}")
    @field_validator('pick_up_location','destination_location',mode= "before")
    @classmethod
    def validate_destination(cls, value):
        geoCodingAPI = GeoCodingAPI()
        geoCoding_destination = geoCodingAPI.get_geocoding(value)
        if geoCoding_destination["status"] == "OK":
            if geoCoding_destination['results'][0]['formatted_address'] == cls.pick_up_location:
                raise ValueError(f"Invalid destination location: {value}")
            else:
                return {"destination_result": geoCoding_destination}
                
        else:
            raise ValueError(f"Invalid destination location: {value}")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # quote_id: str
    # booking_info: BookingCarDetails
    
def check_what_is_empty(user_personal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_personal_details.model_dump().items():
        if value in [None, "", 0]:  # Add other 'empty' conditions if needed
            print(f"Field '{field}' is empty.")
            ask_for.append(field)
    return ask_for


def add_non_empty_details(current_details: BookingCarDetails, new_details: BookingCarDetails):
    non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    updated_details = current_details.model_copy(update=non_empty_details)
    return updated_details


def ask_for_info(ask_list: list):
    first_prompt = ChatPromptTemplate.from_template(
        """Ask one question at a time, even if you don't get all the info. Don't list the questions or greet the user. 
        Explain you're gathering info to help. If 'ask_for' is empty, thank the user and ask how you can assist next.
        ### ask_for list: {ask_for}"""
    )

    info_gathering_chain = first_prompt | llm | StrOutputParser()
    ai_chat = info_gathering_chain.invoke({"ask_for": ask_list})
    print(first_prompt)
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

@tool
def get_booking_details(state : State):
    """ Call function to get the details for a booking from user"""
    chain = llm.with_structured_output(BookingCarDetails)
    print(state["messages"])
    response =chain.invoke(state["messages"])
    booking_details = BookingCarDetails(
        name="", number_phone="", pick_up_location="", destination_location="", pick_up_time=""
    )
    booking_details = add_non_empty_details(booking_details, response)
    ask_for = check_what_is_empty(booking_details)
    ai_response = ask_for_info(ask_for)
    print(ai_response)
    text_input = input()
    user_details, ask_for = filter_response(text_input, booking_details)
    while ask_for:  
        ai_response = ask_for_info(ask_for)
        print(ai_response)
        text_input = input()
        user_details, ask_for = filter_response(text_input, user_details)
        print(ask_for)
    return user_details
@tool
def get_quotes(booking_details : BookingCarDetails):
    """Call function to fetches quotes for car bookings based on the provided booking details."""
    quotesAPI = QuotesAPI(os.getenv("JUPITER_API") + "/demand/v1/quotes")
    geoCodingAPI = GeoCodingAPI()
    print("GeoCoding API")
    # geoCoding_destination =
    geoCoding_pickup =  geoCodingAPI.get_geocoding(booking_details.pick_up_location)
    geoCoding_destination = geoCodingAPI.get_geocoding(booking_details.destination_location)
    
    # input_datetime = datetime.fromisoformat(pick_up_time)
    pickup_datetime = "2025-01-20T09:24:10.000Z"
    
    pickup_coords = { "latitude": float(geoCoding_pickup['results'][0]['geometry']['location']['lat']),"longitude": float(geoCoding_pickup['results'][0]['geometry']['location']['lng']),}
    destination_coords = { "latitude": float(geoCoding_destination['results'][0]['geometry']['location']['lat']),"longitude": float(geoCoding_destination['results'][0]['geometry']['location']['lng']),}
    quotes_data = quotesAPI.get_quotes(pickup_datetime, pickup_coords, destination_coords)
    return quotes_data
@tool
def ask_confirm(booking_details : BookingCarDetails):
    """Ask the user again to confirm the booking details. """
    message = (
        f"Please confirm your ride details:\n"
        f"- Pickup Location: {booking_details.pick_up_location}\n"
        f"- Destination: {booking_details.destination_location}\n"
        f"- Pickup Time: {booking_details.pick_up_time}\n"
        f"- Name: {booking_details.name}\n"
        f"- Contact Number: {booking_details.number_phone}\n"
    )
    print(message)
    user_input = input()
    return "User confirm " + user_input

# def booking_car(state : State):
#     bookingAPI = BookingAPI(bookingsAPI)
#     # quote_id = st"quoteId")
#     person_name = tracker.get_slot("person_name")
#     number_contact = tracker.get_slot("number_contact")
    
#     passenger_info = {
#         "title": "Mr",
#         "phone": number_contact,
#         "firstName": person_name,
#         "lastName": ""
#     }

#     response = bookingAPI.create_booking(
#         quote_id=quote_id,
#         passenger_info=passenger_info
#     )
# memory = MemorySaver()


# graph_builder = StateGraph(State)
 
# llm_with_tools = llm.bind_tools(tools)
# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}
# graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)

# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# # Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

# graph = graph_builder.compile(checkpointer=memory)
tools = [get_booking_details,ask_confirm, get_quotes]  

# config = {"configurable": {"thread_id": "1"}}

# graph_builder = StateGraph(State)
    

system_prompt = """
        You are a very powerful assistant. Be polite, clear, and understandable.
        If users ask general questions, answer them helpfully. If users want to book a ride, 
        call the 'get_booking_details' function to gather booking information.
        If you have full booking information, ask the user to confirm the booking.
        If the user confirms, call the 'get_quotes' function to retrieve quotes. 

    """
agent_executor = create_react_agent(llm, tools = tools, state_modifier=system_prompt)
inputs = {"messages": []}  

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chatbot. Goodbye!")
        break
    inputs["messages"].append(("user", user_input))
    for s in agent_executor.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(f"Assistant: {message[1]}")
        else:
            message.pretty_print()
            