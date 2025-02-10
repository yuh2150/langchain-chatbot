import os
from typing import  Literal 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command, interrupt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.state import State , Router , RouterState ,BookingCarDetails , ConfirmDetails
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from booking_agent.api.booking import BookingAPI
from booking_agent.api.geoCoding import GeoCodingAPI
from booking_agent.api.getKey import OAuthClient
from booking_agent.api.getQuotes import QuotesAPI
from booking_agent.api.is_Airport import IsAirport
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
jupiterAPI = os.getenv('JUPITER_API')
quoteAPI = str(jupiterAPI) + "/demand/v1/quotes"
bookingsAPI  = str(jupiterAPI) + '/demand/v1/bookings'
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant for a car booking system. 
            Your job is to help users book a car. If you don't understand a customer's request, ask questions to clarify.
            Make sure your answers are short but informative.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
class Quote:
    def __init__(self, quote_id, expires_at, vehicle_type, price_value, price_currency, luggage, passengers, provider_name, provider_phone):
        self.quote_id = quote_id
        self.expires_at = expires_at
        self.vehicle_type = vehicle_type
        self.price_value = price_value
        self.price_currency = price_currency
        self.luggage = luggage
        self.passengers = passengers
        self.provider_name = provider_name
        self.provider_phone = provider_phone
    def to_dict(self):
        return {
            "quote_id": self.quote_id,
            "expires_at": self.expires_at,
            "vehicle_type": self.vehicle_type,
            "price_value": self.price_value,
            "price_currency": self.price_currency,
            "luggage": self.luggage,
            "passengers": self.passengers,
            "provider_name": self.provider_name,
            "provider_phone": self.provider_phone
        }
    def __repr__(self):
        return (f"Quote(quote_id={self.quote_id}, expires_at={self.expires_at}, vehicle_type={self.vehicle_type}, "
                f"price_value={self.price_value}, price_currency={self.price_currency}, luggage={self.luggage}, "
                f"passengers={self.passengers}, provider_name={self.provider_name}, provider_phone={self.provider_phone})")
def check_what_is_empty(user_personal_details):
    ask_for = []
    for field, value in user_personal_details.model_dump().items():
        if value in [None, "", 0 ,]:  
            ask_for.append(field)
    return ask_for
def add_non_empty_details(current_details: BookingCarDetails, new_details: BookingCarDetails):
    non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    if new_details.pick_up_location != '': 
        non_empty_details["flight_code"] = new_details.flight_code
    updated_details = current_details.model_copy(update=non_empty_details)
    return updated_details
class NodeUtils:
    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = llm.invoke(prompt)
        return {"messages": response}
    def info_chain(state : State):
        if "booking_info" in state:
            booking_details = state["booking_info"]
        else:
            booking_details = BookingCarDetails(name="", number_phone="", pick_up_location="", destination_location="", pick_up_time="", flight_code= "")
        chain = llm.with_structured_output(BookingCarDetails)
        response = chain.invoke(state["messages"][-1].content)
        user_details = add_non_empty_details(booking_details, response)
        ask_for = check_what_is_empty(user_details)
        return Command(update={"slot_empty": ask_for , "booking_info": user_details})
    def get_quotes(state :State):
        """Call function to fetches quotes for car bookings based on the provided booking details."""
        quotesAPI = QuotesAPI(os.getenv("JUPITER_API") + "/demand/v1/quotes")
        geoCodingAPI = GeoCodingAPI()
        # geoCoding_destination =
        geoCoding_pickup =  geoCodingAPI.get_geocoding(state['booking_info'].pick_up_location)
        geoCoding_destination = geoCodingAPI.get_geocoding(state['booking_info'].destination_location)
        # input_datetime = datetime.fromisoformat(pick_up_time)
        pickup_datetime = "2025-02-10T19:24:10.000Z"
        
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
        return Command(
            update={
                "messages": [
                    {
                        "role": "ai",
                        "content": "chooese Quotes",
                    }
                ]
            },
            # goto=active_agent,
        )

    def human_node(
        state: State, config
    ):
    # -> Command[Literal["get_info"]] :
        """A node for collecting user input."""
        
        user_input = interrupt(value="Ready for user input.")
        return Command(
            update={
                "messages": [
                    {
                        "role": "human",
                        "content": user_input,
                    }
                ]
            },
            goto="get_info"
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
    def get_state(state : State):
        slot_empty = state["slot_empty"]
        if slot_empty == [] :
            return "ask_confirm"
        else : 
            return "ask_info_empty"
    def ask_user_confirm(state : State):
        """Ask the user again to confirm the booking details. """
        message = (
            f"Please confirm your ride details:\n"
            f"- Pickup Location: {state["booking_info"].pick_up_location}\n"
            f"- Destination: {state["booking_info"].destination_location}\n"
            f"- Pickup Time: {state["booking_info"].pick_up_time}\n"
            f"- Name: {state["booking_info"].name}\n"
            f"- Contact Number: {state["booking_info"].number_phone}\n"
        )
        if state["booking_info"].flight_code != 'No Request':
                message += f"- Flight Code: {state['booking_info'].flight_code}\n"
        return Command(update= {"messages": [
                        {
                            "role": "ai",
                            "content": message,
                    }
                ]} )
    def router_node(state: State):
        system_message = "Classify the incoming query as either about booking or not."
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        router_model = llm.with_structured_output(Router)
        route = router_model.invoke(messages)
        return {"route": route["route"]}

    def route_after_prediction(
        state: RouterState,
    ) -> Literal["booking_graph", "call_model"]:
        if state["route"] == "booking":
            return "booking_graph"
        else:
            return "call_model"
    def get_confirm_state(state: State):
        user_confirm = state["messages"][-1].content
        chain_confirm = llm.with_structured_output(ConfirmDetails)
        response = chain_confirm.invoke(user_confirm)
        if (response.confirm == "True") :
            return "get_quotes_booking"
        else :
            if(response.request == "None"):
                return "confirm_change"
                
            elif(response.request == "Cancel"):
                return "cancel_book"
            else:

                return "perform_request"
    def human_node_confirm(
        state: State, config
    ):
        user_confirm = interrupt(value="Please confirm details booking.")
        return Command(
            update={
                "messages": [{"role": "human","content": user_confirm}]
            }
            # goto=active_agent,
        )
    def human_choose_quote(
        state: State, config
    ):
        """A node for collecting user confirm."""
        
        user_confirm = interrupt(value="Please choose quote.")
        command = [Command(
            update={
                "messages": [{"role": "human","content": user_confirm,}]
            }
            # goto=active_agent,
        ),Command(update={"quote_id": user_confirm}) ]
        return command
    def accept_booking(state :State ):
        """Call function to accept booking with quote_ID."""
        bookingAPI = BookingAPI(bookingsAPI)
        # quote_id = tracker.get_slot("quoteId")
        person_name = state["booking_info"].name
        number_contact = state["booking_info"].number_phone
        
        passenger_info = {
            "title": "Mr",
            "phone": number_contact,
            "firstName": person_name,
            "lastName": ""
        }
        print(state["quote_id"])
        response = bookingAPI.create_booking(
            quote_id=state["quote_id"],
            passenger_info=passenger_info
        )
        print(response)
        return response
    def cancel_booking (state : State):
        return Command(
            update={
                "messages": [
                    {
                        "role": "ai",
                        "content": "Your booking has been canceled.",
                    }
                ]
            }
        )

    def confirm_change(state :State):
        return Command (
            update= {
                "messages": [
                    {
                        "role": "ai",
                        "content": "Do you want change booking infomation?",
                    }
                ]
            },
            goto="human_request"
        ) 
        
    def get_change_state(state :State):
        # ["ask_request", "cancel_book","perform_request"]
        
        user_confirm = state["messages"][-1].content
        chain_confirm = llm.with_structured_output(ConfirmDetails)
        response = chain_confirm.invoke(user_confirm)
        
        if (response.confirm == "True"):
            if (response.request == "None"):
                return "ask_request"
            else :
                # updates = [Command(update={"change_request" : response.request}) , "perform_request"]
                return "perform_request"
        else :
            return "cancel_book"
            
    def human_node_request(state :State, config):
        """A node for collecting user confirm."""
        user_confirm = interrupt(value="change details booking.")

        return Command(
            update={
                "messages": [{"role": "human","content": user_confirm}]
            }
            # goto=active_agent,
        )
    def ask_request(state :State):
        
        return Command(
            update={
                "messages": [
                    {
                        "role": "ai",
                        "content": "What would you like to change?",
                    }
                ]
            }
        )
    def human_ans_change(state :State, config):
        """A node for collecting user confirm."""
        user_confirm = interrupt(value="Please answer.")
        chain_confirm = llm.with_structured_output(ConfirmDetails)
        response = chain_confirm.invoke(user_confirm.messages)
        return Command(update={"change_request" : response.request})

    def perform_request(state :State , config) -> Command[Literal["ask_confirm"]] :
        chain = llm.with_structured_output(BookingCarDetails)
        response =chain.invoke(state["messages"][-1].content)
        user_details = add_non_empty_details(state["booking_info"],response)
        ask_for = check_what_is_empty(user_details)
        return Command(
            update={
                "slot_empty": ask_for ,
                "booking_info" : user_details
            },
            goto="get_info",
        ) 