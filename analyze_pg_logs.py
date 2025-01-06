from pathlib import Path

import requests
import tiktoken
import re
import logging
import google.generativeai as genai
import asyncio
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

g_total_input_tokens = 0


def load_prompts(file_path='prompts.txt'):
    prompts = {}
    current_prompt = None
    current_content = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('[') and line.strip().endswith(']'):
                    if current_prompt:
                        prompts[current_prompt] = '\n'.join(current_content).strip()
                    current_prompt = line.strip()[1:-1]
                    current_content = []
                else:
                    current_content.append(line.rstrip())

        if current_prompt:
            prompts[current_prompt] = '\n'.join(current_content).strip()

        return prompts
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {file_path}")
        return None
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
    model = genai.GenerativeModel(model)

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


def send_plan_to_ai(plan, model, timeout):
    static_prompt = g_prompts.get('PLAN_ANALYSIS', '')
    full_prompt = static_prompt + "\n\n" + plan

    return call_ai_provider(full_prompt, model, timeout)


def call_ai_provider(prompt, model, timeout):
    if model.startswith("gpt"):
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
        ]
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


def parse_log_entry_with_title(log_entry):
    # Extract the timestamp from the first 23 characters of the log entry
    timestamp = log_entry[:23].strip()
    # Extract the block from "Query Text:" to "Settings:"
    match = re.search(r"Query Text:(.*?)Settings:", log_entry, re.DOTALL)
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
            <p>{report['chatgpt_hints']}</p>
            <div id="{app_id}"  style="min-height: 400px;">
                <pev2 :plan-source="plan" :plan-query="query"/>
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
        content += f"<tr scope='row'><td>{query_name[:140]} ({query_codes[query_name]})</td><td>{count}</td></tr>"

    content += "</tbody> </table> </div></div>"
    content += f"<p>{frequent_hints_analysis}</p>"

    html = html_template.format(content=content, model=model)
    Path(output_path).write_text(html, encoding="utf-8")


def stable_hash_five_characters(value):
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


from collections import defaultdict


def main(log_file_path, model, output_path, max_ai_calls, timeout):
    model_token_limit = g_token_limits.get(model, 8192)
    in_plan = False
    plan_lines = []
    reports = []
    days = defaultdict(list)  # Initialize days as a defaultdict of lists
    ai_call_count = 1  # Initialize the counter
    last_plan_line = 0  # Initialize the line counter
    query_occurrences = {}  # Map to store query occurrences by title
    query_codes = {}  # Map to store query occurrences by title

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not in_plan and 'plan:' in line:
                in_plan = True
                plan_lines = [line]
                last_plan_line = line_number  # Update the last detected plan line
                continue

            if in_plan:
                plan_lines.append(line)
                if line.strip().startswith("Settings:"):
                    plan_text = "".join(plan_lines)
                    parsed_result = parse_log_entry_with_title(plan_text)

                    query_name = parsed_result["query_name"]
                    title = parsed_result["job_name"] + query_name
                    execution_plan = parsed_result["execution_plan"]
                    estimated_tokens = estimate_token_count(plan_text, model)
                    timestamp = parsed_result["timestamp"]
                    day = timestamp[:10]
                    query_code = stable_hash_five_characters(query_name)

                    logger.debug(f"Estimated tokens for plan: {estimated_tokens}")

                    # Update query occurrences
                    query_occurrences[query_name] = query_occurrences.get(query_name, 0) + 1
                    query_codes[query_name] = query_code

                    if estimated_tokens > model_token_limit:
                        logger.warning(
                            f"Token count ({estimated_tokens}) exceeds the model limit ({model_token_limit}). AI analysis skipped.")
                        ai_hints = f"Token count ({estimated_tokens}) exceeds the model limit ({model_token_limit}). AI analysis skipped."
                    else:
                        try:
                            logger.info(
                                f"Sending plan to AI ({model}) for analysis for query at line {last_plan_line: } (call #{ai_call_count})")
                            ai_hints = send_plan_to_ai(plan_text, model, timeout)
                            ai_call_count += 1  # Increment the counter

                        except Exception as e:
                            ai_hints = f"Error during ChatGPT analysis: {e}"

                    report = {
                        "title": title,
                        "chatgpt_hints": ai_hints,
                        "plan": execution_plan,
                        "query_text": parsed_result["query_text"],
                        "query_timestamp": timestamp,
                        "query_name": query_name,
                        "job_name": parsed_result["job_name"],
                        "code": query_code,
                        "day": day  # Add the day to each report
                    }

                    reports.append(report)
                    days[day].append(report)  # Add the report to the corresponding day

                    in_plan = False
                    plan_lines = []

                    # Exit the loop if AI has been called more than the limit.
                    if max_ai_calls != -1 and ai_call_count > max_ai_calls:
                        logger.info(f"Reached maximum AI calls limit ({max_ai_calls}). Stopping analysis.")
                        break

    analysis = create_analysis(reports, model, timeout)

    generate_html_report(output_path, analysis, model, query_occurrences, days, query_codes)

    logger.info(f"Total input tokens processed: {g_total_input_tokens}")


def create_analysis(reports, model, timeout):
    logger.info("Creating final analysis...")

    # Concatenate all chatgpt_hints
    all_hints = "\n\n".join([report["chatgpt_hints"] for report in reports if report["chatgpt_hints"]])

    # Prepare the prompt for identifying most frequent optimization hints
    prompt_template = g_prompts.get('FINAL_ANALYSIS', '')
    prompt = prompt_template.format(all_hints=all_hints)

    # Call ChatGPT API with the concatenated hints
    return call_ai_provider(prompt, model, timeout)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PostgreSQL log file and generate an analysis report.")
    parser.add_argument("log_filename", help="Path to the PostgreSQL log file")
    parser.add_argument("-m", "--model", default="gemini-2.0-flash-exp",
                        help="AI model to use for analysis (default: gemini-2.0-flash-exp)")
    parser.add_argument("-c", "--max-ai-calls", type=int, default=-1,
                        help="Maximum number of AI calls to make. Use -1 for unlimited (default: -1)")
    parser.add_argument("-t", "--timeout", type=int, default=90,
                        help="Timeout for AI API calls in seconds (default: 90)")
    args = parser.parse_args()

    logger.info(f"Processing PostgreSQL log file {args.log_filename}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Maximum AI calls: {args.max_ai_calls if args.max_ai_calls != -1 else 'Unlimited'}")
    logger.info(f"AI API call timeout: {args.timeout} seconds")
    logger.info(f"Output report: {args.log_filename}_report.html")

    # Load API keys
    api_keys = load_api_keys()
    if api_keys:
        g_openai_key = api_keys.get('openai_key')
        g_gemini_key = api_keys.get('gemini_key')
    else:
        logger.error("Failed to load API keys. Exiting.")
        exit(1)

    g_prompts = load_prompts()

    if not g_prompts:
        logger.error("Failed to load prompts. Exiting.")
        exit(1)

    main(args.log_filename, args.model, f"{args.log_filename}_report.html", args.max_ai_calls, args.timeout)
