
from typing_extensions import TypedDict
from typing import Annotated, Literal , List , Optional , Any
from pydantic import BaseModel, Field , field_validator, ValidationInfo , model_validator
from langgraph.graph.message import AnyMessage , add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
import sys
import os
import requests
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from booking_agent.api.booking import BookingAPI
from booking_agent.api.geoCoding import GeoCodingAPI
from booking_agent.api.getKey import OAuthClient
from booking_agent.api.getQuotes import QuotesAPI
from booking_agent.api.is_Airport import IsAirport
# from booking_agent.val_globals import pick_up_result , destination_result ,booking_details
jupiterAPI = os.getenv('JUPITER_API')
quoteAPI = str(jupiterAPI) + "/demand/v1/quotes"
bookingsAPI  = str(jupiterAPI) + '/demand/v1/bookings'
# booking_agent\val_globals.py
def getData_for_duckling(text, dims):
    url = 'http://localhost:8000/parse'
    data = {
        'locale': 'en_US',
        'text': text,
        'dims': dims,
        'tz': "Asia/Ho_Chi_Minh"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        json_response = response.json()
        # value = json_response[0]['value']['value']
        return json_response
    else:
        return f"Error: {response.status_code}"
pick_up_result = None
destination_result = None
booking_details = None

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
    flight : str = Field(..., description="Request or No Request")
    
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
    @field_validator('pick_up_time')
    @classmethod
    def validate_pick_up_time(cls, value : str):
        dimensions = ["time"]
        if value == '':
            return ''
        
        try:
            expected_format = "%Y-%m-%dT%H:%M:%S.%f%z"
            parsed_datetime = datetime.strptime(value, expected_format)
            return value
        except :
            data = getData_for_duckling(value,dimensions)
            if data and isinstance(data, list) and 'value' in data[0] and 'value' in data[0]['value']:
                return data[0]['value']['value']
            else:
                raise ValueError("Invalid time format")
         
    
    
    @model_validator(mode="after")
    def set_flight_code_if_airport(self)-> str:
        global pick_up_result , destination_result ,booking_details
        geoCodingAPI = GeoCodingAPI()
        API_Airport = IsAirport(base_url=jupiterAPI + '/v2/distance/airport')

        if self.pick_up_location:
            geoCoding_pickup = geoCodingAPI.get_geocoding(self.pick_up_location)
            pick_up_result = geoCoding_pickup
            if geoCoding_pickup["status"] == "OK":
                pick_up_lat = geoCoding_pickup['results'][0]['geometry']['location']['lat']
                pick_up_lng = geoCoding_pickup['results'][0]['geometry']['location']['lng']
                
                is_Airport = API_Airport.is_Airport(pick_up_lat, pick_up_lng)
                if is_Airport[0] == False:  # Nếu là sân bay
                    self.flight = 'No request'
                    self.flight_code = 'None'
                else :
                    self.flight = 'Request'
                    
                    
        if self.destination_location:
            geoCoding_destination = geoCodingAPI.get_geocoding(self.destination_location)
            destination_result = geoCoding_destination
        if booking_details is None:
            booking_details = self
        else:
            non_empty_details = {key: value for key, value in self.dict().items() if value not in [None, ""]}
            booking_details = booking_details.copy(update=non_empty_details)
        
        # print(booking_details)
        return self
def add_non_empty_details(current_details: BookingCarDetails, new_details: BookingCarDetails):
    non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
    updated_details = current_details.model_copy(update=non_empty_details)
    return updated_details
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    quote_id: str
    booking_info: BookingCarDetails 
    slot_empty: list
    change_request : List
    handle_request : str
    
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
        description="User confirmation intent. 'True' if the user confirm booking, 'False' otherwise or want to change information."
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

class FieldChange(BaseModel):
    field_name: str = Field(
        ..., description= "The name of the field to be changed. Must be one of: 'name', 'number_phone', 'pick_up_location', 'destination_location', 'pick_up_time', 'flight_code'."
    )
    new_value: Optional[str] = Field(None, description="The new value of the field,return 'None' if unspecified.")

class ChangeRequest(BaseModel):
    changes: List[FieldChange] = Field(..., description="A list of requested changes, each containing a field name and its new value.")


