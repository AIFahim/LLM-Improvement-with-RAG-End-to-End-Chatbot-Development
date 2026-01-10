"""
Class 07: Multi-Agent Systems - Streamlit UI

Interactive web application for collaborative report writing
using CrewAI multi-agent system.

Features:
- Configure crew and workflow
- Input report topic and requirements
- Monitor agent execution
- View and download generated reports
"""

import streamlit as st
import os
from datetime import datetime
from typing import Optional
import json

from crew_main import ReportWritingCrew, create_report_crew
from crew_agents import get_agent_info, AGENT_CONFIGS
from crew_tasks import get_workflow_info


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Report Writer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    if "current_report" not in st.session_state:
        st.session_state.current_report = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("Configuration")

    # Crew Type Selection
    st.sidebar.subheader("Crew Configuration")

    crew_types = list(AGENT_CONFIGS.keys())
    crew_type = st.sidebar.selectbox(
        "Crew Type",
        crew_types,
        index=0,
        help="Select the type of agent crew to use",
    )

    # Show crew description
    if crew_type in AGENT_CONFIGS:
        st.sidebar.info(AGENT_CONFIGS[crew_type]["description"])

    # Process Selection
    process = st.sidebar.radio(
        "Execution Process",
        ["sequential", "hierarchical"],
        index=0,
        help="Sequential: Tasks run in order. Hierarchical: Manager delegates tasks.",
    )

    # Verbose Mode
    verbose = st.sidebar.checkbox("Verbose Mode", value=True)

    # Memory
    memory = st.sidebar.checkbox("Enable Agent Memory", value=True)

    # Workflow Selection
    st.sidebar.subheader("Workflow")

    workflows = list(get_workflow_info().keys())
    workflow = st.sidebar.selectbox(
        "Workflow Template",
        workflows,
        index=0,
        help="Select the workflow for report generation",
    )

    # Show workflow description
    workflow_info = get_workflow_info()
    if workflow in workflow_info:
        st.sidebar.info(workflow_info[workflow]["description"])

    # Writing Style
    st.sidebar.subheader("Report Settings")

    style = st.sidebar.selectbox(
        "Writing Style",
        ["professional", "academic", "casual", "technical"],
        index=0,
    )

    word_count = st.sidebar.slider(
        "Target Word Count",
        min_value=500,
        max_value=5000,
        value=1500,
        step=100,
    )

    return {
        "crew_type": crew_type,
        "process": process,
        "verbose": verbose,
        "memory": memory,
        "workflow": workflow,
        "style": style,
        "word_count": word_count,
    }


def display_agent_info():
    """Display information about available agents."""
    st.subheader("Available Agents")

    agent_info = get_agent_info()

    cols = st.columns(len(agent_info))

    for idx, (name, info) in enumerate(agent_info.items()):
        with cols[idx]:
            st.markdown(f"**{info['role']}**")
            st.caption(info["description"])
            if info["can_delegate"]:
                st.markdown("*Can delegate tasks*")


def display_execution_progress(placeholder, status: str, agent: str = ""):
    """Display execution progress."""
    with placeholder.container():
        if agent:
            st.info(f"Current Agent: **{agent}**")
        st.progress(0.5)
        st.caption(status)


def display_report(report: dict):
    """Display the generated report."""
    st.subheader("Generated Report")

    if report.get("success"):
        # Report metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Execution Time", f"{report['execution_time']:.2f}s")
        with col2:
            st.metric("Tasks Completed", report.get("tasks_completed", "N/A"))
        with col3:
            st.metric("Agents Used", len(report.get("agents_used", [])))

        # Show agents used
        st.write("**Agents:**", ", ".join(report.get("agents_used", [])))

        st.divider()

        # Report content
        st.markdown(report.get("result", "No content"))

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Report (Markdown)",
                data=report.get("result", ""),
                file_name=f"report_{report['topic'][:20].replace(' ', '_')}.md",
                mime="text/markdown",
            )
        with col2:
            st.download_button(
                label="Download Metadata (JSON)",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"report_{report['topic'][:20].replace(' ', '_')}.json",
                mime="application/json",
            )
    else:
        st.error(f"Report generation failed: {report.get('error', 'Unknown error')}")


def display_history():
    """Display execution history."""
    st.subheader("Execution History")

    if not st.session_state.execution_history:
        st.info("No reports generated yet.")
        return

    for idx, report in enumerate(reversed(st.session_state.execution_history)):
        with st.expander(
            f"{report['topic'][:50]}... - {report['timestamp'][:10]}",
            expanded=(idx == 0),
        ):
            if report.get("success"):
                st.success("Completed successfully")
                st.write(f"**Workflow:** {report['workflow']}")
                st.write(f"**Time:** {report['execution_time']:.2f} seconds")
                st.write(f"**Agents:** {', '.join(report.get('agents_used', []))}")

                if st.button(f"View Report #{idx}", key=f"view_{idx}"):
                    st.session_state.current_report = report
                    st.rerun()
            else:
                st.error(f"Failed: {report.get('error', 'Unknown error')}")


def main():
    """Main application function."""
    init_session_state()

    # Title
    st.title("Multi-Agent Report Writer")
    st.markdown(
        "*Class 07: Multi-Agent Systems - Collaborative Report Writing with CrewAI*"
    )

    # Sidebar configuration
    config = create_sidebar()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Generate Report", "View Report", "History"])

    with tab1:
        st.subheader("Create a New Report")

        # Display agent info
        with st.expander("View Available Agents", expanded=False):
            display_agent_info()

        st.divider()

        # Input form
        with st.form("report_form"):
            topic = st.text_input(
                "Report Topic",
                placeholder="e.g., The Impact of AI on Healthcare",
                help="Enter the main topic for your report",
            )

            requirements = st.text_area(
                "Additional Requirements (Optional)",
                placeholder="e.g., Focus on recent developments, include statistics, target audience is healthcare professionals...",
                help="Any specific requirements or constraints for the report",
            )

            submitted = st.form_submit_button(
                "Generate Report",
                type="primary",
                disabled=st.session_state.is_running,
            )

        if submitted and topic:
            st.session_state.is_running = True

            # Progress placeholder
            progress_placeholder = st.empty()

            with st.spinner("Initializing crew..."):
                # Create crew
                crew = ReportWritingCrew(
                    verbose=config["verbose"],
                    process=config["process"],
                    memory=config["memory"],
                )
                crew.setup_crew(crew_type=config["crew_type"])
                st.session_state.crew = crew

            with st.spinner("Generating report... This may take a few minutes."):
                # Show progress
                progress_placeholder.info(
                    f"Running {config['workflow']} workflow with {config['crew_type']} crew..."
                )

                # Generate report
                result = crew.create_report(
                    topic=topic,
                    requirements=requirements if requirements else None,
                    style=config["style"],
                    word_count=config["word_count"],
                    workflow=config["workflow"],
                )

                # Store results
                st.session_state.current_report = result
                st.session_state.execution_history.append(result)

            st.session_state.is_running = False
            progress_placeholder.empty()

            if result.get("success"):
                st.success("Report generated successfully!")
                st.balloons()
            else:
                st.error(f"Failed: {result.get('error')}")

            # Switch to view tab
            st.rerun()

        elif submitted and not topic:
            st.warning("Please enter a report topic.")

    with tab2:
        if st.session_state.current_report:
            display_report(st.session_state.current_report)
        else:
            st.info("No report to display. Generate a report first.")

    with tab3:
        display_history()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        Multi-Agent Report Writer | Class 07: CrewAI Multi-Agent Systems |
        Planner -> Executor -> Critic Model
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
