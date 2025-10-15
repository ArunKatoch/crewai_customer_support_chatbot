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
    def kickoff_flow(self):
        """Initialize the customer support flow"""
        print("\n=== Customer Support Ticket System ===\n")
        return self.state

    @listen(kickoff_flow)
    def get_input(self):
        """Step 0: Get input from the user about the product and issue"""
        print("--- Step 0: Collecting Ticket Information ---\n")

        # Get user input
        self.state.product_name = input("What product would you like to support for? ")
        # Get issue description with validation
        while not self.state.issue_description:
            self.state.issue_description = input("What issue are you facing? ")

        print(f"\nProcessing ticket for {self.state.product_name}...")
        return self.state

    @listen(get_input)
    def validate_ticket(self):
        """Step 1: Validate the customer support ticket"""
        print("\n--- Step 1: Validating Ticket ---\n")

        # Initialize the customer support crew
        customer_support_crew = CustomerSupportCrew()

        # Get only the validator agent and task
        validator_agent = customer_support_crew.ticket_validator()
        validation_task = customer_support_crew.validate_ticket_task()

        # Set up the task inputs
        validation_task.description = validation_task.description.format(
            product_name=self.state.product_name,
            issue_description=self.state.issue_description
        )
        validation_task.agent = validator_agent

        # Create a crew with just validation
        from crewai import Crew, Process
        validation_crew = Crew(
            agents=[validator_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True
        )

        # Run validation
        result = validation_crew.kickoff()

        # Extract validation result
        if result.pydantic:
            self.state.ticket_validated = result.pydantic.valid_ticket
            print(f"\nTicket Valid: {self.state.ticket_validated}\n")

        return self.state

    @listen(validate_ticket)
    def classify_ticket(self):
        """Step 2: Classify the ticket type and subject"""
        print("\n--- Step 2: Classifying Ticket ---\n")

        # Initialize the customer support crew
        customer_support_crew = CustomerSupportCrew()

        # Get only the classifier agent and task
        classifier_agent = customer_support_crew.ticket_classifier()
        classification_task = customer_support_crew.classify_ticket_task()

        # Set up the task inputs
        classification_task.description = classification_task.description.format(
            product_name=self.state.product_name,
            issue_description=self.state.issue_description
        )
        classification_task.agent = classifier_agent

        # Create a crew with just classification
        from crewai import Crew, Process
        classification_crew = Crew(
            agents=[classifier_agent],
            tasks=[classification_task],
            process=Process.sequential,
            verbose=True
        )

        # Run classification
        result = classification_crew.kickoff()

        # Extract classification results
        if result.pydantic:
            self.state.ticket_type = result.pydantic.ticket_type
            self.state.ticket_subject = result.pydantic.ticket_subject
            self.state.reasoning = result.pydantic.reasoning
            print(f'\nClassification: {self.state.ticket_type} - {self.state.ticket_subject}')
            print(f'Reasoning: {self.state.reasoning}\n')

        return self.state

    @listen(classify_ticket)
    def resolve_ticket(self):
        """Step 3: Generate resolution for the ticket"""
        print("\n--- Step 3: Generating Resolution ---\n")

        # Initialize the customer support crew
        customer_support_crew = CustomerSupportCrew()

        # Get only the resolver agent and task
        resolver_agent = customer_support_crew.ticket_resolver()
        resolution_task = customer_support_crew.resolve_ticket_task()

        # Set up the task inputs
        resolution_task.description = resolution_task.description.format(
            product_name=self.state.product_name,
            issue_description=self.state.issue_description,
            ticket_type=self.state.ticket_type,
            ticket_subject=self.state.ticket_subject
        )
        resolution_task.agent = resolver_agent

        # Create a crew with just resolution
        from crewai import Crew, Process
        resolution_crew = Crew(
            agents=[resolver_agent],
            tasks=[resolution_task],
            process=Process.sequential,
            verbose=True
        )

        # Run resolution
        result = resolution_crew.kickoff()

        # Extract resolution
        self.state.final_resolution = result.raw

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
