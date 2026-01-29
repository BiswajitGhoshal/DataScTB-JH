import streamlit as st
from io import StringIO
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import pandas as pd
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool

os.environ["nonOPEN"] = ""
llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.environ["OPENAI_API_KEY"]
        )

class CSVReaderTool(BaseTool):
    name: str = "CSVReaderTool"
    description: str = "read and return a given number of records from a given feedback type"
    def _run(self, df_name: str, recno_start: int, recno_end: int) -> pd.DataFrame:
        df = pd.read_csv(df_name)
        return(df[recno_start:recno_end])

class TicketWriterTool(BaseTool):
    name: str = "TicketWriterTool"
    description: str = "write the ticket details received in json format into a csv file after converting it into a dataframe"
    def _run(self, json_tkt: str):
        df = pd.read_json(json_tkt)
        df.to_csv("tickets.csv")

csv_reader_agent = Agent(
    role='Read CSV data',
    goal='Reads and parses feedback data from CSV files',
    backstory='Expert in reading a CSV file and returning the required records',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

feedback_classifier_agent = Agent(
    role='Classify feedback into one of the categories: bug, feature request, praise, complaint and spam data',
    goal='Categorize feedback into given categories',
    backstory='Expert in understanding customer issues from their feedback and categorizing those',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

bug_analysis_agent = Agent(
    role='Extract technical details: steps to reproduce, platform info, severity assessment - and output those in json format',
    goal='Extract technical details from feedback text provided it is classified as a bug',
    backstory='Expert in finding technical details from customer feedback',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

feature_extractor_agent = Agent(
    role='Identifies new feature requests and estimates user impact/demand from user feedback - and output those in json format',
    goal='Identify new feature requests from feedback text provided the feedback is classified as feature request',
    backstory='Expert in identifying feature requests from customer feedback',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

ticket_creactor_agent = Agent(
    role='Creates a list output containing the source_id, source_type, category, priority, technical_details and suggested_title using the outputs from other agents',
    goal='Create ticket details from feedback text',
    backstory='Expert in creating ticket details from customer feedback',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

quality_critic_agent = Agent(
    role='Reviews whether the ticket details produced by ticket_creator_agent are really present in the feedback record',
    goal='Ensure ticket details present in the feedback record',
    backstory='Expert in reviewing ticket details from customer feedback record',
    verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

csv_reader_tool = CSVReaderTool()
ticket_writer_tool = TicketWriterTool()
#finance_tool = YahooFinanceTool()

csv_read_task = Task(
    description="read the {input_file} file, and return records between {start_rec_no} and {end_rec_no}.",
    expected_output="JSON formatted records.",
    agent=csv_reader_agent,
    tools=[csv_reader_tool]
)

feedback_classifier_task = Task(
    description="identify the category of the feedback.",
    expected_output="A string containing one of - bug, feature request, praise, complaint, spam.",
    agent=feedback_classifier_agent
)

bug_analysis_task = Task(
    description="extracts technical details from a feedback, provided it is categorized as a bug by feedback_classifier_task.",
    expected_output="A json containing identified technical details.",
    agent=bug_analysis_agent
)

feature_extractor_task = Task(
    description="extracts features requested from a feedback, provided it is categorized as a feature request by feedback_classifier_task.",
    expected_output="A list containing identified features.",
    agent=feature_extractor_agent
)

ticket_creator_task = Task(
    description="Generates structured tickets and logs them to output CSV file named {out_file}.",
    expected_output="A list containing the ticket information.",
    agent=ticket_creactor_agent,
    tools=[ticket_writer_tool]
)

quality_critic_task = Task(
    description="Reviews generated tickets against feedback record and finds discrepancies.",
    expected_output="A string containing review comment.",
    agent=quality_critic_agent
)

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Feedback Processor",
    layout="wide")

st.title("Feedback Processing Dashboard")

# -----------------------
# Mock data sources
# (replace with os.listdir or DB queries in real app)
# -----------------------
feedback_input_types = [
    "app_feedback_samples.csv",
    "support_emails.csv"
]

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Configuration")

    st.subheader("Feedback Source Selection")

    # Feedback input file selector
    selected_type = st.selectbox(
        "Feedback input - Type",
        feedback_input_types
    )

    start_rec_no = st.number_input("Enter start record number:", format="%.0f")
    end_rec_no = st.number_input("Enter end record number:", format="%.0f")

    st.write("Entered numbers are:", int(start_rec_no), " and ", int(end_rec_no), " respectively.")

    st.divider()

    # Output file name
    output_file_name = st.text_input(
        "Output File Name",
        value="processed_tickets.csv"
    )

    run_clicked = st.button("Go", type="primary")

# -----------------------
# Main panel layout
# -----------------------
col1, col2 = st.columns(2)

# -----------------------
# Panel 1 – Data Output
# -----------------------
with col1:
    st.subheader("Processed Feedback")
    panel_1 = st.container(height=420)

    with panel_1:
        if run_clicked:
            crew = Crew(
            agents=[csv_reader_agent,feedback_classifier_agent, bug_analysis_agent, feature_extractor_agent,ticket_creactor_agent,quality_critic_agent],
            tasks=[csv_read_task, feedback_classifier_task, bug_analysis_task, feature_extractor_task, ticket_creator_task, quality_critic_task],
            verbose=False  # Set to False to reduce rich console output and avoid RecursionError
            )

            input_params = {
                'input_file': selected_type,
                'start_rec_no': start_rec_no,
                'end_rec_no': end_rec_no,
                'out_file': str(output_file_name)
            }
#            result = FoodCatalyst().crew().kickoff(inputs=inputs)

            result = crew.kickoff(inputs=input_params)
            print("\n📊 Ticket Creation Report:\n")
            print(result)

        else:
            st.info("Click **Go** to view processed feedback data.")

# -----------------------
# Panel 2 – Logs / Text Output
# -----------------------
with col2:
    st.subheader("Processing Logs & Summary")
    panel_2 = st.container(height=420)

    with panel_2:
        if run_clicked:
            log_text = f"""
            Processing started...
            Source (Type 1): {selected_type}

            Steps completed:
            - Loaded feedback files
            - Applied filters
            - Aggregated results
            - Generated output file

            Processing completed successfully.
            """
            st.text_area(
                label="Logs",
                value=log_text,
                height=340
            )
            # Mock processed dataframe
            df = pd.DataFrame({
                "Feedback ID": range(1, 11),
                "Category": ["Positive", "Negative", "Neutral"] * 3 + ["Positive"],
                "Source": [selected_type] * 10,
                "Comment": [f"Feedback comment {i}" for i in range(1, 11)]
            })

            st.dataframe(df, use_container_width=True)

        else:
            st.info("Logs will appear here after processing.")

# -----------------------
# Download Output File
# -----------------------
if run_clicked:
    # Convert dataframe to CSV for download
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Output File",
        data=csv_buffer.getvalue(),
        file_name=output_file_name,
        mime="text/csv"
    )

    st.success("Output file ready for download.")