#!/usr/bin/env python

import json
from random import randint
from typing import Literal
from pydantic import BaseModel, Field
from crewai.flow import Flow, listen, start
from crewai import LLM,Agent,Task,Crew
import os
from dotenv import load_dotenv

load_dotenv()

from deep_resarch.crews.poem_crew.poem_crew import PoemCrew

unique_ticket_types = ['technical_issue', 'billing_inquiry', 'cancellation_request', 'product_inquiry', 'refund_request']
unique_ticket_subset = ['network_problem', 'account_access', 'data_loss', 'software_bug', 'product_setup', 'product_recommendation', 'cancellation_request', 'product_compatibility', 'hardware_issue', 'delivery_problem', 'payment_issue', 'refund_request', 'display_issue', 'battery_life', 'installation_support', 'peripheral_compatibility']




class TicketType(BaseModel):
    ticket_type: Literal[*unique_ticket_types] = Field(..., description="The type of ticket")
    ticket_subject: Literal[*unique_ticket_subset] = Field(..., description="The subset of ticket")
    reasoning: str = Field(..., description="Brief explanation for the classification")



class CustomerSupportState(BaseModel):
    product_name:str = ""
    issue_description:str = ""
    ticket_type:str = ""
    ticket_subject:str = ""
    reasoning:str = ""
    final_resolution:str = ""
    ticket_validated:bool = False

class ParseAndValidateTicket(BaseModel):
    valid_ticket:bool = True
    
class CustomerSupportFlow(Flow[CustomerSupportState]):
    
    @start()
    def get_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== Create Your Comprehensive Guide ===\n")

        # Get user input
        self.state.product_name = input("What product would you like to support for? ")
        # Get audience level with validation
        while not self.state.issue_description:
            self.state.issue_description = input("What issue are you facing").lower()

        print(self.state)
        return self.state

    @listen(get_input)
    def parse_and_validate_ticket(self):
        """Parse and validate the ticket"""
        validation_llm = LLM(model=os.getenv("MODEL"),temperature=0.0,response_format = ParseAndValidateTicket)
        message = [
            {"role": "system", "content": "You are a helpful assistant designed to Check and validate the ticket.Provide a boolean value if the ticket is valid or not."},
            {"role": "user", "content": f"Ticket: {self.state.issue_description} and product name: {self.state.product_name}"}
        ]
        response = validation_llm.call(message)
        response = json.loads(response)
        self.state.ticket_validated = response['valid_ticket']
        return self.state


    @listen(parse_and_validate_ticket)
    def generate_type_and_subject(self):
        """Generate the type and subject of the ticket"""
        type_and_subject_llm = LLM(model=os.getenv("MODEL"),temperature=0.0)
        ticket_classifier_agent = Agent(
            role='Customer Support Ticket Classifier',
            goal='Accurately classify customer support ticket descriptions into the correct ticket type',
            backstory="""You are an expert customer support analyst with years of experience 
            in categorizing and triaging customer support tickets. You have deep understanding 
            of different types of customer issues including technical problems, billing inquiries, 
            cancellation requests, product inquiries, and refund requests. You analyze ticket 
            descriptions carefully and classify them accurately based on the customer's actual need.""",
            verbose=True,
            allow_delegation=False,
            llm=type_and_subject_llm
        )
        
        def create_classification_task(ticket_description: str):
            """Create a task to classify a ticket description"""
            
            # Create a formatted list of possible ticket types
            ticket_types_list = "\n".join([f"- {tt}" for tt in unique_ticket_types])
            ticket_subset_list = "\n".join([f"- {tt}" for tt in unique_ticket_subset])
            
            
            task = Task(
                description=f"""Analyze the following customer support ticket description and classify it 
                into one of the available ticket types.
                
                Ticket Description:
                {ticket_description}
                
                Available Ticket Types:
                {ticket_types_list}

                Available Ticket subject:
                {ticket_subset_list}
                
                Carefully read the ticket description and determine which ticket type best matches 
                the customer's issue or request. Consider the main intent and primary concern of the customer.
                """,
                expected_output="""A JSON object with the following structure:
                {{
                    "ticket_type": "the classified ticket type (must be one from the available types)",
                    "ticket_subject": "the classified ticket subject (must be one from the available subjects)",
                    "reasoning": "brief explanation of why you chose this classification"
                }}""",
                agent=ticket_classifier_agent,
                output_pydantic=TicketType
            )
        classification_task = create_classification_task(self.state.issue_description)
        # Create crew
        crew = Crew(
            agents=[ticket_classifier_agent],
            tasks=[classification_task],
            verbose=False
        )
        
        # Run the crew
        result = crew.kickoff()

        response = result.pydantic
        self.state.ticket_type = response.ticket_type
        self.state.ticket_subject = response.ticket_subject
        self.state.reasoning = response.reasoning
        return self.state

    @listen(generate_type_and_subject)
    def generate_resolution(self):
        return self.state



def kickoff():
    customer_support_flow = CustomerSupportFlow()
    customer_support_flow.kickoff()


def plot():
    customer_support_flow = CustomerSupportFlow()
    customer_support_flow.plot()


if __name__ == "__main__":
    kickoff()
