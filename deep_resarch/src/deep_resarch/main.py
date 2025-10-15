#!/usr/bin/env python

import json
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv

load_dotenv()

from deep_resarch.crews.customer_support_crew.customer_support_crew import CustomerSupportCrew


class CustomerSupportState(BaseModel):
    product_name: str = ""
    issue_description: str = ""
    ticket_type: str = ""
    ticket_subject: str = ""
    reasoning: str = ""
    final_resolution: str = ""
    ticket_validated: bool = False
    
class CustomerSupportFlow(Flow[CustomerSupportState]):

    @start()
    def get_input(self):
        """Get input from the user about the product and issue"""
        print("\n=== Customer Support Ticket System ===\n")

        # Get user input
        self.state.product_name = input("What product would you like to support for? ")
        # Get issue description with validation
        while not self.state.issue_description:
            self.state.issue_description = input("What issue are you facing? ")

        print(f"\nProcessing ticket for {self.state.product_name}...")
        return self.state

    @listen(get_input)
    def process_ticket(self):
        """Process the customer support ticket using the CustomerSupportCrew"""
        print("\n--- Processing Ticket ---\n")

        # Initialize the customer support crew
        customer_support_crew = CustomerSupportCrew()

        # Prepare inputs for the crew
        inputs = {
            'product_name': self.state.product_name,
            'issue_description': self.state.issue_description,
            'ticket_type': '',  # Will be filled by classify task
            'ticket_subject': '',  # Will be filled by classify task
        }

        # Run the crew
        result = customer_support_crew.crew().kickoff(inputs=inputs)

        # Extract results from the crew execution
        # The last task output contains the resolution
        self.state.final_resolution = result.raw

        # Get classification results from the classify task
        if len(result.tasks_output) >= 2:
            classify_output = result.tasks_output[1]
            if classify_output.pydantic:
                self.state.ticket_type = classify_output.pydantic.ticket_type
                self.state.ticket_subject = classify_output.pydantic.ticket_subject
                self.state.reasoning = classify_output.pydantic.reasoning
                print(f'\nClassification: {self.state.ticket_type} - {self.state.ticket_subject}')
                print(f'Reasoning: {self.state.reasoning}\n')

        # Get validation result
        if len(result.tasks_output) >= 1:
            validation_output = result.tasks_output[0]
            if validation_output.pydantic:
                self.state.ticket_validated = validation_output.pydantic.valid_ticket

        print("\n--- Resolution ---")
        print(self.state.final_resolution)
        print("\n")

        return self.state



def kickoff():
    customer_support_flow = CustomerSupportFlow()
    customer_support_flow.kickoff()


def plot():
    customer_support_flow = CustomerSupportFlow()
    customer_support_flow.plot()


if __name__ == "__main__":
    kickoff()
