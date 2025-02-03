from langchain_openai import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini") 

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
def check_what_is_empty(user_peronal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_peronal_details.model_dump().items():
        if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for
def add_non_empty_details(current_details: BookingCarDetails, new_details: BookingCarDetails):
    non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    updated_details = current_details.model_copy(update=non_empty_details)
    return updated_details

def ask_for_info(ask_list:list):
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        """Ask one question at a time, even if you don't get all the info. Don't list the questions or greet the user. 
        Explain you're gathering info to help. If 'ask_for' is empty, thank the user and ask how you can assist next.
        ### ask_for list: {ask_for}"""
    )
    # info_gathering_chain
    info_gathering_chain = first_prompt | llm | StrOutputParser()
    ai_chat = info_gathering_chain.invoke({"ask_for": ask_list})
    print(first_prompt)
    return ai_chat
def filter_response(text_input, user_details ):
    chain = llm.with_structured_output(BookingCarDetails)
    res = chain.invoke(text_input)
    # add filtered info to the
    user_details = add_non_empty_details(user_details,res)
    print(user_details)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for
def get_booking_details(response) :
    booking_details = BookingCarDetails(name="",number_phone="",pick_up_location="",destination_location="", pick_up_time="")
    booking_details = add_non_empty_details(booking_details ,response)
    ask_for = check_what_is_empty(booking_details)
    
    while ask_for:  
        if ask_for:
            ai_response = ask_for_info(ask_for)
            print(ai_response)
        else:
            print('Everything gathered move to next phase')
        text_input = input()
        user_details, ask_for = filter_response(text_input, user_details)

    # tools = [get_booking_details]
    system_prompt = """
    You are a very powerful assistant. Be polite, clear, and understandable.
    If users ask general questions, answer them helpfully. If users want to book a ride, 
    call the 'get_booking_details' function to gather booking information, and pass the user's final input to the function.
    Remember to keep the user's last input
    """
    agent_executor = create_react_agent(llm, tools = [get_booking_details] , state_modifier=system_prompt)
    inputs = {"messages": [("user", "my name is Huy,I want to book a car from 271 Nguyen Van Linh to 466 NGuyen van linh da nang at now")]}
    for s in agent_executor.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()