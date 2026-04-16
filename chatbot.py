"""
Simple AI chatbot wrapper using Groq (free tier).
Get your free API key at: https://console.groq.com
"""

import os
import glob
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
TOKEN_LIMIT = 5000  # safe limit under Groq's 6000 TPM for free tier

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

    # Binary / non-text extensions to skip when loading folders
    SKIP_EXT = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
        ".woff", ".woff2", ".ttf", ".eot",
        ".zip", ".tar", ".gz", ".br",
        ".pyc", ".pyo", ".so", ".dylib", ".dll",
        ".db", ".sqlite", ".sqlite3",
        ".lock",
    }
    SKIP_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".venv", ".next", "dist", "build"}

    def load_file(self, path: str) -> str | None:
        path = os.path.expanduser(os.path.abspath(path))
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", errors="replace") as f:
                content = f.read()
        except (PermissionError, IsADirectoryError):
            return None
        self.loaded_files.append(path)
        rel = os.path.basename(path)
        self.history.append({
            "role": "user",
            "content": f"I'm loading this file for reference — {rel} ({path}):\n\n```\n{content}\n```"
        })
        self.history.append({
            "role": "assistant",
            "content": f"Got it. I've read {rel} ({len(content.splitlines())} lines). Ask me anything about it."
        })
        return path

    def load_path(self, path: str) -> list[str]:
        """Load a file, folder, or glob pattern. Returns list of loaded paths."""
        path = os.path.expanduser(path.strip())

        # Handle glob patterns (e.g. *.ts, **/*.py)
        if any(c in path for c in ("*", "?")):
            matches = sorted(glob.glob(path, recursive=True))
            loaded = []
            for m in matches:
                if os.path.isfile(m) and self._should_load(m):
                    result = self.load_file(m)
                    if result:
                        loaded.append(result)
            return loaded

        path = os.path.abspath(path)

        # Single file
        if os.path.isfile(path):
            result = self.load_file(path)
            return [result] if result else []

        # Directory — walk and load text files
        if os.path.isdir(path):
            loaded = []
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    if self._should_load(fpath):
                        result = self.load_file(fpath)
                        if result:
                            loaded.append(result)
            return loaded

        return []

    def _should_load(self, path: str) -> bool:
        _, ext = os.path.splitext(path)
        return ext.lower() not in self.SKIP_EXT

    def _estimate_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(m["content"]) for m in self.history)
        return total_chars // 4

    def chat(self, user_message: str) -> str:
        self.turn += 1
        self.history.append({"role": "user", "content": user_message})

        est = self._estimate_tokens()
        if est > TOKEN_LIMIT:
            self.history.pop()  # remove the message we just added
            self.turn -= 1
            return (
                f"Context too large (~{est:,} tokens, limit ~{TOKEN_LIMIT:,}). "
                f"Use /clear to reset, or load fewer files."
            )

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
  {YELLOW}/load{RESET}           — interactive mode: paste paths one per line
  {YELLOW}/load <paths>{RESET}  — inline: comma-separated files, folders, or globs
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

        if user_input.lower() == "/loaded":
            if not bot.loaded_files:
                print(f"{CYAN}BorisBot:{RESET} {GRAY}No files loaded yet.{RESET}\n")
            else:
                files = "\n    ".join(bot.loaded_files)
                print(f"{CYAN}BorisBot:{RESET} Files in context:\n    {YELLOW}{files}{RESET}\n")
            continue

        if user_input.lower().startswith("/load"):
            arg = user_input[5:].strip()
            paths = []
            if arg:
                # Inline: /load path1, path2
                paths = [p.strip() for p in arg.split(",") if p.strip()]
            else:
                # Interactive: paste paths one per line, empty line to finish
                print(f"{GRAY}Paste paths one per line (empty line to finish):{RESET}")
                while True:
                    try:
                        line = input(f"  {GRAY}path:{RESET} ").strip()
                    except (KeyboardInterrupt, EOFError):
                        break
                    if not line:
                        break
                    paths.append(line)
            all_loaded = []
            for path in paths:
                print(f"{GRAY}Loading {path}...{RESET}")
                loaded = bot.load_path(path)
                if not loaded:
                    print(f"  {YELLOW}Not found or empty:{RESET} {path}")
                else:
                    all_loaded.extend(loaded)
                    for f in loaded:
                        print(f"  {GREEN}✓{RESET} {f}")
            if all_loaded:
                print(f"{CYAN}BorisBot:{RESET} {GREEN}Loaded {len(all_loaded)} file(s).{RESET} Ask me anything about them.\n")
            elif paths:
                print(f"{CYAN}BorisBot:{RESET} {YELLOW}No files loaded.{RESET}\n")
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
