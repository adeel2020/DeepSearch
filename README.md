# DeepResearchAgent

## Project Logic Overview

DeepResearchAgent is a modular, agent-based AI system for automated, high-quality research and report generation. The workflow is orchestrated by specialized agents, each with clear responsibilities and handoffs, as described in the code's docstrings and agent instructions:

- **Requirement Gathering:** The RequirementGatheringAgent interacts with the user to clarify and collect research requirements, using interactive prompts if needed.
- **Planning:** The PlanningAgent generates a step-by-step plan based on user requirements before passing control to the main research workflow.
- **Orchestration:** The LeadResearchAgent manages the research process, delegating tasks to:
  - WebSearchAgent for deep, multi-perspective web research
  - SythesisAgent for fact-checking, citation, and source reliability assessment
  - ReportWriterAgent for professional, structured report writing
- **Synthesis & Verification:** The SythesisAgent coordinates fact-finding, citation, and source-checking sub-agents. It ensures all facts are verified, cited, and any disagreements or uncertainties are clearly highlighted.
- **Report Writing:** The ReportWriterAgent produces a comprehensive, well-structured, and cited report, using a formal and professional style.
- **Personalization:** User profiles are used to tailor research instructions and examples.

The system emphasizes reliability, transparency, and clarity by:
- Spawning multiple searches for deep research
- Highlighting source reliability (High/Medium/Low)
- Detecting and reporting conflicting information between sources
- Providing inline citations and references in all outputs

---

## Features

- **Automated Deep Research:** Gathers and summarizes information on any topic using web search and LLMs.
- **Agent-Oriented Design:** Specialized agents for requirement gathering, planning, research, synthesis, fact-checking, and report writing.
- **Source Quality & Conflict Detection:** Rates sources by reliability and highlights conflicting information between sources.
- **Citations & References:** All outputs include inline citations and references.
- **Personalized Research:** Supports user profiles and research preferences.
- **Extensible:** Easily add new agents or tools for custom workflows.

---

## Project Structure

```
deep_research_system.py      # Main orchestrator and entry point
planning_agent.py            # Planning and requirement gathering agents
research_agents.py           # Web search, fact-checking, and source quality agents
synthesis_agent.py           # Synthesis and citation agents
report_writer.py             # Professional report writing agent
.env                         # API keys and environment variables
pyproject.toml / uv.lock     # Project dependencies (managed by uv)
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd DeepResearchAgent
   ```

2. **Install [uv](https://github.com/astral-sh/uv):**
   ```sh
   pip install uv
   # or, for Homebrew users:
   brew install uv
   ```

3. **Install dependencies with uv:**
   ```sh
   uv pip install -r requirements.txt
   # or, if you prefer, use:
   uv pip install .
   ```

4. **Configure API keys:**
   - Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `TAVILY_API_KEY`.

---

## Usage

### Run the Full Research System

```sh
uv run planning_agent.py
```
- This will prompt you for a research topic and guide you through the full workflow: requirement gathering, planning, research, synthesis, and report writing.

### Run Standalone Agents

- **Web Search Agent:**  
  ```sh
  uv run research_agents.py
  ```
- **Synthesis Agent:**  
  ```sh
  uv run synthesis_agent.py
  ```
- **Report Writer:**  
  ```sh
  uv run report_writer.py
  ```

---

## Configuration

- **API Keys:**  
  Set your API keys in the `.env` file:
  ```
  OPENAI_API_KEY=...
  GEMINI_API_KEY=...
  TAVILY_API_KEY=...
  ```

- **Python Version:**  
  Requires Python 3.12 or higher.

- **Dependency Management:**  
  This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. All commands should use `uv` instead of `pip` or `venv`.

---

## Dependencies
- [uv](https://github.com/astral-sh/uv)
- [openai-agents](https://pypi.org/project/openai-agents/)
- [tavily-python](https://pypi.org/project/tavily-python/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [pydantic](https://pypi.org/project/pydantic/)

---

## Example Workflow

1. **User provides a research topic.**
2. **Requirement gathering agent** clarifies and collects user requirements.
3. **Planning agent** breaks down the research task.
4. **Web search agent** gathers information.
5. **Fact-checking and source checker agents** verify facts, rate source reliability, and detect conflicts.
6. **Synthesis agent** summarizes findings and adds citations.
7. **Report writer agent** generates a professional, cited report.

---

## License

MIT License

---

*For questions or contributions, please open an issue or pull request.*
