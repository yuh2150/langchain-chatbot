
from typing_extensions import TypedDict
from typing import Annotated, Literal , List
from pydantic import BaseModel, Field , field_validator, ValidationInfo , model_validator
from langgraph.graph.message import AnyMessage , add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from booking_agent.api.booking import BookingAPI
from booking_agent.api.geoCoding import GeoCodingAPI
from booking_agent.api.getKey import OAuthClient
from booking_agent.api.getQuotes import QuotesAPI
from booking_agent.api.is_Airport import IsAirport
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
    flight_code: str = Field(
        # default= 'None',
        ...,
        description="Flight numbers, consisting of letters and numbers, usually start with the airline code (e.g. VN123, SQ318)."
    )
    
    @field_validator('pick_up_location')
    @classmethod
    def validate_pickup(cls, value:str, info: ValidationInfo):
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
    @model_validator(mode="after")
    def set_flight_code_if_airport(self):
        """ Kiểm tra nếu điểm đón là sân bay thì gán flight_code = '123' """
        geoCodingAPI = GeoCodingAPI()
        API_Airport = IsAirport(base_url=jupiterAPI + '/v2/distance/airport')

        if self.pick_up_location:
            geoCoding_pickup = geoCodingAPI.get_geocoding(self.pick_up_location)
            if geoCoding_pickup["status"] == "OK":
                pick_up_lat = geoCoding_pickup['results'][0]['geometry']['location']['lat']
                pick_up_lng = geoCoding_pickup['results'][0]['geometry']['location']['lng']
                
                is_Airport = API_Airport.is_Airport(pick_up_lat, pick_up_lng)

                if is_Airport[0] == False:  # Nếu là sân bay
                    self.flight_code = 'No Request'
        return self

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    quote_id: str
    booking_info: BookingCarDetails 
    slot_empty: list
    change_request : str
    
class Router(TypedDict):
    route: Literal["booking", "other"]
    
class RouterState(MessagesState):
    route: Literal["booking", "other"]
from pydantic import BaseModel, Field

class ConfirmDetails(BaseModel):
    """
    Represents the user's confirmation intent and request details.
    """
    confirm: str = Field(
        ..., 
        description="User confirmation intent. True if the user confirms booking, False otherwise."
    )
    request: str = Field(
        ..., 
        description="""Return request related to name, pickup and destination location, pickup time, phone number ,flight code.
        Returns 'None' if there is no request.Returns 'Cancel' if there is cancel."""
    )
    # continue_booking: bool = Field(
    #     # default= 'True',
    #     description="True if the user wants to change info and continue booking, False if they want to cancel."
    # )