import streamlit as st
from io import StringIO
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import pandas as pd
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool

os.environ["OPENAI_API_KEY"] = ""

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Utilization Processor",
    layout="wide")

st.title("Utilization Reporting Dashboard (Capability-demo)")

utilization_input_types = [
    "utilization.csv"
]

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Configuration")

    st.subheader("Utilization Source Selection")

    # Feedback input file selector
    selected_type = st.selectbox(
        "Feedback input - Type",
        utilization_input_types
    )

    if selected_type == "utilization.csv":
        st.text("5 records. Enter numbers between 1 and 5")

    start_rec_no = st.number_input("Enter start record number:", format="%.0f")
    end_rec_no = st.number_input("Enter end record number:", format="%.0f")
    st_rec_no = 0

    if start_rec_no != 0:
        st_rec_no = int(start_rec_no) - 1

    user_q = st.text_input("Enter your quesiton:", "which team has lowest average utilization?")

    st.write("Records numbers to be used are: ", st_rec_no+1, " and ", int(end_rec_no), " and user_question is : ", user_q)

    temp_api_key = st.text_input("Enter ChatGPT key to be used: ")
    os.environ["OPENAI_API_KEY"] = temp_api_key

    llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.environ["OPENAI_API_KEY"]
        )

    class CSVReaderTool(BaseTool):
        name: str = "CSVReaderTool"
        description: str = "read and return a given number of records from selected data source"
        def _run(self, df_name: str, recno_start: int, recno_end: int) -> pd.DataFrame:
            df = pd.read_csv(df_name)
            return(df[recno_start:recno_end])

    class SuggestionWriterTool(BaseTool):
        name: str = "SuggestionWriterTool"
        description: str = "write the suggestion into a text file  answering the question"
        def _run(self, json_tkt: str, out_file: str):
            df = pd.read_json(json_tkt)
            df.to_csv(out_file)

    class MetricsWriterTool(BaseTool):
        name: str = "MetricsWriterTool"
        description: str = "write the metrics details received in string format into a text file"
        def _run(self, texte: str, out_file: str):
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(texte)


    csv_reader_agent = Agent(
        role='Read CSV data',
        goal='Reads and parses feedback data from CSV files',
        backstory='Expert in reading a CSV file and returning the required records',
        verbose=True,  # Keep agent verbose for debugging, we'll adjust Crew verbose
        allow_delegation=False,
        llm=llm
    )

    suggestion_creactor_agent = Agent(
        role='Answers the given question {user_q} based on the records selected',
        goal='Analyze the records and answer the question in {user_q} write the final answer into {out_file} only once',
        backstory='Expert in analyzing team utilization data',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    quality_critic_agent = Agent(
        role='Reviews whether the answer produced by suggestion_creator_agent are relevant for the records and output quality metrics',
        goal='Ensure the answer is relatable to the records and evaluate overall crew performance',
        backstory='Expert in reviewing answer to the question {user_q} from the given utilization records and calculate metrics',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    csv_reader_tool = CSVReaderTool()
    suggestion_writer_tool = SuggestionWriterTool()
    metrics_writer_tool = MetricsWriterTool()

    csv_read_task = Task(
        description="read the {input_file} file, and return records between {st_rec_no} and {end_rec_no}.",
        expected_output="JSON formatted records.",
        agent=csv_reader_agent,
        tools=[csv_reader_tool]
    )

    answer_creator_task = Task(
        description="Generates answer to the question {user_q} asked by analyzing the records and then output CSV file named {out_file} - try only one time.",
        expected_output="A file containing the generated response.",
        agent=suggestion_creactor_agent,
        tools=[suggestion_writer_tool]
    )

    quality_critic_task = Task(
        description="Reviews generated answer to the question {user_q} against the records and finds discrepancies.",
        expected_output="A file named 'metrics.txt' giving total number of records processed, average confidence_score, number of records with discrepancies and overall comment.",
        agent=quality_critic_agent,
        tools=[metrics_writer_tool]
    )

    st.divider()

    # Output file name
    output_file_name = st.text_input(
        "Output File Name",
        value="generated_suggestion.csv"
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
            agents=[csv_reader_agent, suggestion_creactor_agent, quality_critic_agent],
            tasks=[csv_read_task, answer_creator_task, quality_critic_task],
            verbose=False  # Set to False to reduce rich console output and avoid RecursionError
            )

            input_params = {
                'input_file': selected_type,
                'st_rec_no': st_rec_no,
                'end_rec_no': int(end_rec_no),
                'user_q': user_q,
                'out_file': str(output_file_name)
            }
            tdf = pd.read_csv(selected_type)
            st.text("Selected records are: ")
            st.dataframe(tdf.iloc[int(st_rec_no):int(end_rec_no), :])
            result = crew.kickoff(inputs=input_params)
            print("\n📊 Utilization Response:\n")
            print(result)

        else:
            st.info("Click **Go** to view feedback data being processed.")

# -----------------------
# Panel 2 – Logs / Text Output
# -----------------------
with col2:
    st.subheader("Processing Logs & Summary")
    panel_2 = st.container(height=420)

    with panel_2:
        if run_clicked:
            with st.status("Agent is working...", expanded=True) as status:
                st.write(result)
                status.update(label="Done!", state="complete", expanded=False)
            st.markdown(result)

            with open('processing_log.txt', 'w') as f:
                f.write(str(result.tasks_output)+str(result.token_usage))
        else:
            st.info("Logs will appear here after processing.")

# -----------------------
# Download Output File
# -----------------------
if run_clicked:
    # Convert dataframe to CSV for download

    csv_buffer = StringIO()
    df = pd.read_csv(output_file_name)
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Generated Suggestions File",
        data=csv_buffer.getvalue(),
        file_name=output_file_name,
        mime="text/csv"
    )

    st.success("Output file ready for download.")