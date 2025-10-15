from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List, Literal
from pydantic import BaseModel, Field
import os


# Pydantic models for structured outputs
class ParseAndValidateTicket(BaseModel):
    valid_ticket: bool = True


class TicketType(BaseModel):
    ticket_type: Literal[
        'technical_issue',
        'billing_inquiry',
        'cancellation_request',
        'product_inquiry',
        'refund_request'
    ] = Field(..., description="The type of ticket")
    ticket_subject: Literal[
        'network_problem',
        'account_access',
        'data_loss',
        'software_bug',
        'product_setup',
        'product_recommendation',
        'cancellation_request',
        'product_compatibility',
        'hardware_issue',
        'delivery_problem',
        'payment_issue',
        'refund_request',
        'display_issue',
        'battery_life',
        'installation_support',
        'peripheral_compatibility'
    ] = Field(..., description="The subset of ticket")
    reasoning: str = Field(..., description="Brief explanation for the classification")


@CrewBase
class CustomerSupportCrew:
    """Customer Support Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self.serper_tool = SerperDevTool()

    @agent
    def ticket_validator(self) -> Agent:
        validation_llm = LLM(
            model=os.getenv("MODEL"),
            temperature=0.0,
            response_format=ParseAndValidateTicket
        )
        return Agent(
            config=self.agents_config["ticket_validator"],
            llm=validation_llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def ticket_classifier(self) -> Agent:
        classifier_llm = LLM(
            model=os.getenv("MODEL"),
            temperature=0.0
        )
        return Agent(
            config=self.agents_config["ticket_classifier"],
            llm=classifier_llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def ticket_resolver(self) -> Agent:
        return Agent(
            config=self.agents_config["ticket_resolver"],
            tools=[self.serper_tool],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def validate_ticket_task(self) -> Task:
        return Task(
            config=self.tasks_config["validate_ticket"],
            output_pydantic=ParseAndValidateTicket,
        )

    @task
    def classify_ticket_task(self) -> Task:
        return Task(
            config=self.tasks_config["classify_ticket"],
            output_pydantic=TicketType,
        )

    @task
    def resolve_ticket_task(self) -> Task:
        return Task(
            config=self.tasks_config["resolve_ticket"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Customer Support Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
