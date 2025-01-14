class BookingForm :
    def __init__(self):
        self.required_slots = ["name","number-phone","pickup_location", "destination_location", "pickup_time"]
        self.slots = {slot: None for slot in self.required_slots}

    def get_next_slot(self):
        for slot, value in self.slots.items():
            if value is None: 
                return slot
        return None 

    def handle_user_input(self, slot, user_input):
        self.slots[slot] = user_input

    def is_complete(self):
        return all(value is not None for value in self.slots.values())

    def run(self):
        print("Welcome! I need some information to book your ride.")

        while not self.is_complete():
            next_slot = self.get_next_slot()
            if next_slot:
                # Hỏi người dùng về slot tiếp theo
                user_input = input(f"Please provide your {next_slot.replace('_', ' ')}: ")
                self.handle_user_input(next_slot, user_input)

        print("Thank you! Here is the information you provided:")
        for slot, value in self.slots.items():
            print(f"{slot.replace('_', ' ').capitalize()}: {value}")
