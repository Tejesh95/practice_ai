````markdown
# Agentic Tool Use Demo with Google Gemini

This project demonstrates how to build an agentic workflow using the Google Gemini API with function calling (tool use). It showcases multiple tool invocations, partial failure handling, and an iterative agent loop that allows the model to reason over tool outputs before producing a final response.

## Features

- Tool/function calling with Google Gemini
- Multiple tool calls in a single interaction
- Parallel tool execution workflow
- Partial failure handling (successful and failed tools together)
- Agentic execution loop
- Automatic generation of a detailed `output.md` report
- Environment variable support using `.env`

## Project Structure

```text
.
├── .env                 # API key (not committed)
├── .gitignore
├── main.py              # Main application
├── output.md            # Generated after execution
├── requirements.txt
└── README.md
```

## Tools Implemented

### Weather Tool
Returns simulated weather information for supported cities.

### Stock Price Tool
Intentionally throws an exception to demonstrate partial failure handling.

### Calculator Tool
Safely evaluates basic arithmetic expressions.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key
```

## Run

```bash
python main.py
```

## Example Prompt

```
What's the weather in Hyderabad and Bangalore, what is the current stock price of GOOG, and also calculate 45 * 12 + 7?
```

## Example Workflow

1. User submits a request.
2. Gemini determines which tools are needed.
3. Multiple tool calls are requested.
4. Python executes each tool.
5. Successful results and failures are sent back to the model.
6. Gemini generates the final response using all available information.

## Partial Failure Handling

One of the tools (`get_stock_price`) is intentionally designed to fail.

Instead of stopping execution:

- Successful tool results are preserved.
- Failed tool calls return structured error information.
- The model receives both successes and failures.
- The final response is generated using the available data.

This demonstrates robust tool orchestration and graceful error handling.

## Output

Running the project generates an `output.md` report containing:

- Tool calls requested
- Tool arguments
- Success/failure status
- Tool outputs
- Error messages
- Final model response
- Summary statistics

## Technologies Used

- Python 3
- Google Gemini API
- google-generativeai
- python-dotenv

## Notes

- The weather data is simulated.
- The stock price tool intentionally fails to demonstrate error handling.
- The calculator supports basic arithmetic operations only.

## License

This project is intended for educational and demonstration purposes.
````
