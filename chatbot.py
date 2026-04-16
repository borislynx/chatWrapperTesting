"""
Simple AI chatbot wrapper using Groq (free tier).
Get your free API key at: https://console.groq.com
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Terminal colors
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are BorisBot, Boris's personal AI assistant. 
You run on the llama-3.1-8b-instant model hosted by Groq.
You specialise in coding and tech questions but can handle anything.
You are direct, no-fluff, and occasionally witty. Keep answers concise."""


class BorisBot:
    def __init__(self, api_key: str, model: str = MODEL):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self.turn = 0
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.loaded_files = []  # tracks which files are in context

    def load_file(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return None
        with open(path, "r", errors="replace") as f:
            content = f.read()
        filename = os.path.basename(path)
        self.loaded_files.append(filename)
        # inject the file as a user message so it lives in conversation history
        self.history.append({
            "role": "user",
            "content": f"I'm loading this file for reference — {filename}:\n\n```\n{content}\n```"
        })
        self.history.append({
            "role": "assistant",
            "content": f"Got it. I've read {filename} ({len(content.splitlines())} lines). Ask me anything about it."
        })
        return filename

    def chat(self, user_message: str) -> str:
        self.turn += 1
        self.history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.turn = 0
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


def print_help():
    print(f"""
{BOLD}Available commands:{RESET}
  {YELLOW}/joke{RESET}          — ask BorisBot for a joke
  {YELLOW}/model{RESET}         — show the current model info
  {YELLOW}/clear{RESET}         — clear conversation history
  {YELLOW}/atlas{RESET}         — show helpful commands for Atlas & Unity
  {YELLOW}/load <path>{RESET}   — load a file into context, then ask questions about it
  {YELLOW}/loaded{RESET}        — list files currently loaded in context
  {YELLOW}/help{RESET}          — show this menu
  {YELLOW}quit{RESET}           — exit
""")


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()

    bot = BorisBot(api_key)

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════╗
║        BorisBot v1.0         ║
║  Your personal AI assistant  ║
╚══════════════════════════════╝{RESET}

{GREEN}Hey Boris! BorisBot here — specialising in code & tech.
Ask me anything. Type {YELLOW}/help{GREEN} for commands.{RESET}
""")

    while True:
        try:
            user_input = input(f"{GRAY}[Turn {bot.turn + 1}]{RESET} {BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}BorisBot:{RESET} Goodbye Boris. 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(f"\n{CYAN}BorisBot:{RESET} Goodbye Boris. 👋\n")
            break

        if user_input.lower() == "/clear":
            bot.reset()
            print(f"{CYAN}BorisBot:{RESET} {GRAY}Conversation cleared. Fresh start.{RESET}\n")
            continue

        if user_input.lower() == "/model":
            print(f"{CYAN}BorisBot:{RESET} Running {YELLOW}{bot.model}{RESET} via Groq.\n")
            continue

        if user_input.lower().startswith("/load "):
            path = user_input[6:].strip()
            print(f"{GRAY}Loading {path}...{RESET}")
            result = bot.load_file(path)
            if result is None:
                print(f"{CYAN}BorisBot:{RESET} {YELLOW}File not found:{RESET} {path}\n")
            else:
                print(f"{CYAN}BorisBot:{RESET} {GREEN}Loaded {result}.{RESET} Ask me anything about it.\n")
            continue

        elif user_input.lower() == "/loaded":
            if not bot.loaded_files:
                print(f"{CYAN}BorisBot:{RESET} {GRAY}No files loaded yet.{RESET}\n")
            else:
                files = "\n    ".join(bot.loaded_files)
                print(f"{CYAN}BorisBot:{RESET} Files in context:\n    {YELLOW}{files}{RESET}\n")
            continue

        elif user_input.lower() == "/joke":
            user_input = "Tell me a short, original joke."

        elif user_input.lower() == "/atlas":
            print(f"""
{CYAN}{BOLD}Atlas & Unity — Helpful Commands:{RESET}

  {YELLOW}Rebuild Unity local:{RESET}
    ./rebuild_unity_local.sh

  {YELLOW}Run dev server:{RESET}
    doppler run --project unity --config dev_personal -- .venv/bin/python manage.py runserver localhost:8166

  {YELLOW}Run pre-commit checks:{RESET}
    pre-commit run --all-files

  {YELLOW}Run tests:{RESET}
    make run-tests
""")
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        reply = bot.chat(user_input)
        print(f"{GRAY}[Turn {bot.turn}]{RESET} {CYAN}{BOLD}BorisBot:{RESET} {reply}\n")


if __name__ == "__main__":
    main()
