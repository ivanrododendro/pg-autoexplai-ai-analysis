#!/usr/bin/env python

from pathlib import Path

import html
import requests
import tiktoken
import re
import logging
import google.generativeai as genai
import asyncio
import hashlib
from collections import defaultdict
import argparse

__QUERY_NAME_LIMIT = 140
__DEFAULT_TOKEN_LIMIT = 8192
__DEFAULT_MODEL_TEMPERATURE = 0.5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

g_model_temperature = __DEFAULT_MODEL_TEMPERATURE
g_model_token_limit = __DEFAULT_TOKEN_LIMIT
g_openai_key = None
g_gemini_key = None
g_prompts = {}
g_total_input_tokens = 0
g_token_limits = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "gemini-2.0-flash-exp": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemini-1.5-pro": 2097152
}


def load_prompts(lang):
    prompts = {}
    current_prompt = None
    current_content = []
    base_path = Path(__file__).parent / 'prompts'

    lang_file_path = f"{base_path}_{lang}.txt"

    try:
        with open(lang_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    if current_prompt:
                        prompts[current_prompt] = '\n'.join(current_content).strip()
                    current_prompt = line[1:-1]
                    current_content = []
                else:
                    current_content.append(line)

        if current_prompt:
            prompts[current_prompt] = '\n'.join(current_content).strip()

        return prompts
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {lang_file_path}")
    except Exception as e:
        logger.error(f"Error reading prompts file: {e}")

    return None


def load_api_keys(file_path='api_keys.txt'):
    keys = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                keys[key] = value
        return keys
    except FileNotFoundError:
        logger.error(f"API keys file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading API keys file: {e}")
        return None


# Add this function to estimate token count
def estimate_token_count(text, model="gpt-4"):
    global g_total_input_tokens
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(text))
    g_total_input_tokens += token_count
    return token_count


async def call_gemini(full_prompt, model, api_key, timeout):
    # Configure the Gemini API
    genai.configure(api_key=api_key)

    # Set up the model
    model = genai.GenerativeModel(model, generation_config=genai.GenerationConfig(temperature=g_model_temperature))

    try:
        # Generate content
        response = await asyncio.wait_for(
            model.generate_content_async(full_prompt),
            timeout=timeout,
        )

        if response.text:
            return response.text
        else:
            logger.warning("No analysis content found in Gemini response.")
            return "No analysis content found in Gemini response."
    except asyncio.exceptions.TimeoutError as e:
        logger.error(f"Timeout while communicating with Gemini API: {e}")
        return None
    except Exception as e:
        logger.error(f"Error communicating with Gemini API: {e}")
        return None


def call_ai_for_plan_analysis(plan, model, timeout):
    static_prompt = g_prompts.get('PLAN_ANALYSIS', '')
    full_prompt = static_prompt + "\n\n" + plan

    return call_ai_provider(full_prompt, model, timeout)


def call_ai_provider(prompt, model, timeout):
    estimated_tokens = estimate_token_count(prompt, model)

    if estimated_tokens > g_model_token_limit:
        ai_hints = f"Token count ({estimated_tokens}) exceeds the model limit ({g_model_token_limit}). AI analysis skipped."
        return None

    if model.startswith("gpt") or model.startswith("o1"):
        return call_chatgpt(prompt, model, g_openai_key, timeout)
    elif model.startswith("gemini"):
        return asyncio.run(call_gemini(prompt, model, g_gemini_key, timeout))
    else:
        logger.error(f"Unsupported model: {model}")
        return None


def call_chatgpt(full_prompt, model, openai_key, timeout=90):
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a PostgreSQL optimization expert."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": g_model_temperature  # Add the temperature parameter here
    }

    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    # Send the POST request to OpenAI API
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers,
                                 verify=False, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        responseJson = response.json()

        if 'choices' in responseJson and len(responseJson['choices']) > 0:
            responseText = responseJson['choices'][0]['message']['content']
        else:
            logger.warning("No analysis content found in ChatGPT response.")
            responseText = "No analysis content found in ChatGPT response."

        return responseText
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error: The request to OpenAI API timed out after {timeout} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with OpenAI API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return None


def parse_log_entry(log_entry):
    # Extract the timestamp from the first 23 characters of the log entry
    timestamp = log_entry[:23].strip()
    # Extract the block from "Query Text:" to "Settings:"
    match = re.search(r"Query Text:(.*?)$", log_entry, re.DOTALL)
    if not match:
        raise ValueError("Could not parse log entry: Missing query text or execution plan.")

    # Extract the full block
    full_block = match.group(1).strip()
    lines = full_block.splitlines()

    # Extract the title (first one or two lines of the query)
    if len(lines) >= 2 and lines[0].strip().startswith("--") and lines[1].strip().startswith("--"):
        job_name = lines[0]
        query_name = lines[1]
        start_index = 2  # Skip the title lines for further processing
    else:
        job_name = ""
        query_name = [lines[0]]
        start_index = 0

    job_name = "\n".join(job_name).strip().replace("\t", "").replace("\n", "")
    query_name = "\n".join(query_name).strip().replace("\t", "").replace("\n", "")

    # Split based on the first occurrence of "cost="
    query_lines = []
    plan_lines = []
    found_plan = False

    for line in lines[start_index:]:  # Start from the appropriate index
        if not found_plan and "cost=" in line:
            found_plan = True
        if found_plan:
            plan_lines.append(line)
        else:
            query_lines.append(line)

    if not plan_lines:
        raise ValueError("No execution plan found in the log entry.")

    return {
        "timestamp": timestamp,
        "query_name": query_name,
        "job_name": job_name,
        "query_text": "\n".join(query_lines).strip(),
        "execution_plan": "\n".join(plan_lines).strip()
    }


# Function to generate an HTML report
def generate_html_report(output_path, frequent_hints_analysis, model, query_occurrences, days, query_codes):
    logger.info(f"Generating HTML report in {output_path}")
    """
    Generates an HTML report based on the provided analysis reports.

    Parameters:
    - reports (list): A list of dictionaries containing title, ChatGPT hints, and execution plans.
    - output_path (str): The path to save the generated HTML file.
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PostgreSQL Auto Explain AI ({model}) Report</title>
    <script src="https://unpkg.com/vue@3.2.45/dist/vue.global.prod.js"></script>
    <script src="https://unpkg.com/pev2/dist/pev2.umd.js"></script>
    <link href="https://unpkg.com/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
   	<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://unpkg.com/pev2/dist/style.css" />
    </head>
    <body class="container-fluid">
        <script>
            const {{ createApp }} = Vue;
        </script>
        <h1 class="mb-4">PostgreSQL Auto Explain AI ({model}) Report</h1>
        <h2>Requêtes</h2>
        {content}
    </body>
    </html>
    """
    content = ""
    sorted_days = sorted(days.keys())

    for day in sorted_days:
        content += f"""
            <a data-toggle="collapse" href="#collapseDay-{day}" role="button" aria-expanded="false" aria-controls="collapseDay-{day}">
                <h3>{day}</h3>
            </a>
            <div class="collapse" id="collapseDay-{day}">
            <div class="card card-body">
        """

        for i, report in enumerate(days[day]):
            # Generate unique IDs for each Vue app instance
            app_id = f"app-{day}-{i}"
            content += f"""
            <a data-toggle="collapse" href="#collapseExample-{app_id}" role="button" aria-expanded="false" aria-controls="collapseExample-{app_id}">
            <h5>{report['query_timestamp']} : {report['title']} ({report['code']})</h5>
            </a>
            <div class="collapse" id="collapseExample-{app_id}">
            <div class="card card-body">
            {report['chatgpt_hints']}
            <div id="{app_id}"  style="min-height: 400px;">
                <pev2 :plan-source="plan" :plan-query="query"></pev2>
            </div>
            <script>
                createApp({{
                    data() {{
                        return {{
                            plan: `{report['plan']}`,
                            query: `{report['query_text']}`
                        }};
                    }}
                }}).component("pev2", pev2.Plan).mount("#{app_id}");
            </script>
            </div>
            </div>      
            """
        content += "</div></div>"

    content += "<h2>Synthèse</h2>"
    content += """
        <a data-toggle="collapse" href="#requestCollapse" role="button" aria-expanded="false" aria-controls="requestCollapse">
            <h3>Requêtes</h3>
        </a>
          <div class="collapse" id="requestCollapse">
        <div class="card card-body">
            <table class='table-striped' >
                <thead>
                    <tr>
                        <th scope='col'>Requête</th><th scope='col'># occurrences</th>
                    </tr>
                </thead>
                <tbody>
                """
    for (query_name, count) in query_occurrences.items():
        content += f"<tr scope='row'><td>{query_name[:__QUERY_NAME_LIMIT]} ({query_codes[query_name]})</td><td>{count}</td></tr>"

    content += "</tbody> </table> </div></div>"
    content += f"{frequent_hints_analysis}"

    html = html_template.format(content=content, model=model)
    Path(output_path).write_text(html, encoding="utf-8")


def hash_five_characters(value):
    """
    Generates a unique and consistent 5-character hash for a given value.

    :param value: The value to hash (string or number).
    :return: A 5-character hash string.
    """
    # Convert the value to a string and encode it
    value_str = str(value).encode('utf-8')

    # Generate a stable hash using SHA-256
    hash_object = hashlib.sha256(value_str)

    # Convert the hash to base36 (numbers and uppercase letters)
    base36 = ""
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    hash_value = int(hash_object.hexdigest(), 16)  # Convert hex to int
    while hash_value > 0:
        hash_value, remainder = divmod(hash_value, 36)
        base36 = alphabet[remainder] + base36

    # Ensure the hash is exactly 5 characters long
    return base36[:5].zfill(5)  # Pad with leading zeros if necessary


def call_ai_for_final_analysis(reports, model, timeout):
    logger.info("Creating final analysis...")

    # Concatenate all chatgpt_hints
    all_hints = "\n\n".join([report["chatgpt_hints"] for report in reports if report["chatgpt_hints"]])

    # Prepare the prompt for identifying most frequent optimization hints
    prompt_template = g_prompts.get('FINAL_ANALYSIS', '')
    prompt = prompt_template.format(all_hints=all_hints)

    # Call ChatGPT API with the concatenated hints
    return call_ai_provider(prompt, model, timeout)


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Process PostgreSQL log file and generate an analysis report.")
    parser.add_argument("log_filename", help="Path to the PostgreSQL log file")
    parser.add_argument("-m", "--model", default="gemini-2.0-flash-exp",
                        help="AI model to use for analysis (default: gemini-2.0-flash-exp)")
    parser.add_argument("-c", "--max-ai-calls", type=int, default=-1,
                        help="Maximum number of AI calls to make. Use -1 for unlimited (default: -1)")
    parser.add_argument("-t", "--timeout", type=int, default=90,
                        help="Timeout for AI API calls in seconds (default: 90)")
    parser.add_argument("-l", "--lang", default="fr",
                        help="Language for prompts and output (default: fr)")
    parser.add_argument("-p", "--temperature", type=float, default=__DEFAULT_MODEL_TEMPERATURE,
                        help="Temperature for the AI model (default: 0.5)")
    return parser.parse_args()


def process_log_file(log_file_path, model, max_ai_calls, timeout):
    reports, days, query_occurrences, query_codes = [], defaultdict(list), {}, {}
    ai_call_count = 1

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if 'plan:' in line:
                plan_lines = extract_plan_lines(f, line)
                parsed_result = parse_log_entry("".join(plan_lines))

                logger.info(
                    f"Sending plan to AI ({model}) for analysis for query at line {line_number} (call #{ai_call_count})")

                report = process_parsed_result(parsed_result, model, timeout)

                if report is not None:
                    ai_call_count += 1

                if report:
                    query_name = report["query_name"]
                    reports.append(report)
                    days[report["day"]].append(report)
                    query_occurrences[query_name] = query_occurrences.get(query_name, 0) + 1
                    query_codes[query_name] = report["code"]

                if max_ai_calls != -1 and ai_call_count > max_ai_calls:
                    break

    return reports, days, query_occurrences, query_codes


def extract_plan_lines(file, first_line):
    plan_lines = [first_line]
    for line in file:
        plan_lines.append(line)
        if line.strip().startswith("Settings:"):
            break
    return plan_lines


def process_parsed_result(parsed_result, model, timeout):
    query_name = parsed_result["query_name"]
    title = parsed_result["job_name"] + query_name
    execution_plan = parsed_result["execution_plan"]
    timestamp = parsed_result["timestamp"]
    day = timestamp[:10]
    query_code = hash_five_characters(query_name)
    query = html.escape(parsed_result["query_text"])

    ai_hints = call_ai_for_plan_analysis(execution_plan, model, timeout)

    report = {
        "title": title,
        "chatgpt_hints": ai_hints,
        "plan": execution_plan,
        "query_text": query,
        "query_timestamp": timestamp,
        "query_name": query_name,
        "job_name": parsed_result["job_name"],
        "code": query_code,
        "day": day
    }

    return report


def main():
    args = parse_cli_arguments()

    logger.info(f"Processing PostgreSQL log file {args.log_filename}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Maximum AI calls: {args.max_ai_calls if args.max_ai_calls != -1 else 'Unlimited'}")
    logger.info(f"AI API call timeout: {args.timeout} seconds")
    logger.info(f"Language: {args.lang}")
    logger.info(f"Output report: {args.log_filename}_report.html")
    logger.info(f"Model temperature : {args.temperature}")

    global g_prompts,  g_model_token_limit, g_model_temperature,  g_openai_key, g_gemini_key

    g_prompts = load_prompts(args.lang)

    if not g_prompts:
        logger.error(f"Failed to load prompts for language: {args.lang}. Exiting.")
        exit(1)

    api_keys = load_api_keys()

    if api_keys:
        g_openai_key = api_keys.get('openai_key')
        g_gemini_key = api_keys.get('gemini_key')
    else:
        logger.error("Failed to load API keys. Exiting.")
        exit(1)

    g_model_token_limit = g_token_limits.get(args.model, __DEFAULT_TOKEN_LIMIT)
    g_model_temperature = args.temperature

    reports, days, query_occurrences, query_codes = process_log_file(
        args.log_filename, args.model, args.max_ai_calls, args.timeout
    )

    analysis = call_ai_for_final_analysis(reports, args.model, args.timeout)

    generate_html_report(f"{args.log_filename}_report.html", analysis, args.model, query_occurrences, days,
                         query_codes)

    logger.info(f"Total input tokens processed: {g_total_input_tokens}")


if __name__ == "__main__":
    main()
