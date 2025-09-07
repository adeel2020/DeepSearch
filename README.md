# DeepResearchAgent

DeepResearchAgent is an AI-powered research orchestration system that automates deep research, fact-checking, and professional report writing using advanced LLMs (Gemini, OpenAI) and web search APIs. It uses a modular, agent-based architecture to break down complex research tasks into specialized steps, ensuring high-quality, cited, and well-structured outputs.

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
pyproject.toml / requirements.txt  # Project dependencies
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd DeepResearchAgent
   ```

2. **Set up Python environment:**
   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   # or, if using poetry:
   poetry install
   ```

4. **Configure API keys:**
   - Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `TAVILY_API_KEY`.

---

## Usage

### Run the Full Research System

```sh
python planning_agent.py
```
- This will prompt you for a research topic and guide you through the full workflow: requirement gathering, planning, research, synthesis, and report writing.

### Run Standalone Agents

- **Web Search Agent:**  
  ```sh
  python research_agents.py
  ```
- **Synthesis Agent:**  
  ```sh
  python synthesis_agent.py
  ```
- **Report Writer:**  
  ```sh
  python report_writer.py
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

---

## Dependencies

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
