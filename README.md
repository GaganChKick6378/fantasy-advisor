# Fantasy IPL Advisor

A Python-based interactive advisor for Fantasy IPL that provides insights, player statistics, and strategic recommendations using AI and data analysis.

## Features

- Interactive Q&A system for IPL-related queries
- Real-time data fetching and analysis
- Vector-based semantic search for relevant information
- Confidence scoring for responses
- User feedback collection
- LangSmith integration for tracing and analytics

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Exa API key
- LangSmith account (optional, for tracing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ipl-new.git
cd ipl-new
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
EXA_API_KEY=your_exa_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional
```

## Project Structure

```
ipl-new/
├── advisor/           # Core advisor implementation
├── data/             # Data storage and processing
├── main.py           # Main application entry point
├── requirements.txt  # Project dependencies
└── .env             # Environment variables (create this)
```

## Usage

1. Ensure your virtual environment is activated and all dependencies are installed.

2. Run the main application:
```bash
python main.py
```

3. The interactive session will start, and you can:
   - Ask questions about IPL players, stats, or strategies
   - Type 'stats' to see vector store statistics
   - Type 'exit' to quit
   - Provide feedback on responses (1-5 rating)

## Features in Detail

- **Interactive Q&A**: Ask any IPL-related questions and get AI-powered responses
- **Confidence Scoring**: Each response comes with a confidence score
- **Context Sources**: Responses include relevant data sources
- **Session Analytics**: View summary statistics after each session
- **LangSmith Integration**: Track and analyze interactions (if configured)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- OpenAI for the language model
- Exa for data fetching capabilities
- LangSmith for tracing and analytics 