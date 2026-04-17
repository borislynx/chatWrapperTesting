"""
Simple AI chatbot wrapper using Groq (free tier).
Get your free API key at: https://console.groq.com
"""

import os
import re
import sys
import glob
import json
import math
import time
import select
import termios
import tty
import tempfile
import subprocess
import datetime
import threading
from openai import OpenAI
from dotenv import load_dotenv

# Optional: voice input (speech-to-text)
try:
    import sounddevice as sd
    import numpy as np
    import soundfile as sf
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


load_dotenv()

# Terminal colors
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

MODEL = "llama-3.1-8b-instant"
TOKEN_LIMIT = 5000  # safe limit under Groq's 6000 TPM for free tier

SYSTEM_PROMPT = """You are BorisBot, Boris's personal AI assistant.
You run on the llama-3.1-8b-instant model hosted by Groq.
You specialise in coding and tech questions but can handle anything.
You are direct and no-fluff. Keep answers concise.
You have tools available: get_current_time and calculate. Use them when relevant.
When a tool has already returned a result, just state it plainly (e.g. "4" or "2*2 = 4"). Do not add commentary or observations about the result."""

# --- Tool Definitions (Agentic AI / Function Calling) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression like '2 + 2' or 'sqrt(144)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
]


TOOL_TEXT_PATTERN = re.compile(r'<function=(\w+)>(.*?)</function>', re.DOTALL)


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "get_current_time":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name == "calculate":
        try:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
            return str(eval(args["expression"], {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"


def parse_text_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse and execute tool calls the model wrote as text instead of proper tool_calls."""
    matches = TOOL_TEXT_PATTERN.findall(text)
    if not matches:
        return []
    results = []
    for name, args_str in matches:
        args = json.loads(args_str) if args_str.strip() else {}
        result = execute_tool(name, args)
        results.append((name, result))
    return results


class BorisBot:
    def __init__(self, api_key: str, model: str = MODEL):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self.turn = 0
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.loaded_files = []
        self.temperature = 0.5   # lower = more direct, higher = more creative (0.0–2.0)
        self.speak_enabled = False
        self.listen_enabled = False
        self._listen_thread = None
        self._chat_lock = threading.Lock()
        self.timestamps = []  # (turn, role, iso_timestamp)

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
        total_chars = sum(len(m.get("content", "") or "") for m in self.history)
        return total_chars // 4

    def chat(self, user_message: str) -> str:
        with self._chat_lock:
            return self._chat_inner(user_message)

    def _chat_inner(self, user_message: str) -> str:
        self.turn += 1
        self.history.append({"role": "user", "content": user_message})
        self.timestamps.append((self.turn, "user", datetime.datetime.now().isoformat()))

        est = self._estimate_tokens()
        if est > TOKEN_LIMIT:
            self.history.pop()
            self.turn -= 1
            return (
                f"Context too large (~{est:,} tokens, limit ~{TOKEN_LIMIT:,}). "
                f"Use /clear to reset, or load fewer files."
            )

        # Try with tools; if Groq rejects (hallucinated tool name), retry without
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=TOOLS,
                temperature=self.temperature,
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
            )
        message = response.choices[0].message

        # Handle proper tool_calls from the API
        if message.tool_calls:
            self.history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_call.function.name, args)
                print(f"  {GRAY}[tool: {tool_call.function.name} -> {result}]{RESET}")
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
            )
            message = response.choices[0].message

        reply = message.content or ""

        # Handle text-based tool calls (model writes <function=name>args</function> as text)
        text_tools = parse_text_tool_calls(reply)
        if text_tools:
            for name, result in text_tools:
                print(f"  {GRAY}[tool: {name} -> {result}]{RESET}")
            tool_summary = ", ".join(f"{name} = {result}" for name, result in text_tools)
            self.history.append({"role": "assistant", "content": reply})
            self.history.append({
                "role": "user",
                "content": f"[Tool results: {tool_summary}] State the result directly, no commentary.",
            })
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    temperature=self.temperature,
                )
                reply = response.choices[0].message.content or ""
            except Exception:
                reply = tool_summary
            # Clean up the injected messages, keep only the final reply
            self.history.pop()  # remove injected user prompt
            self.history.pop()  # remove raw tool-call text

        self.history.append({"role": "assistant", "content": reply})
        self.timestamps.append((self.turn, "assistant", datetime.datetime.now().isoformat()))

        return reply

    # --- Feature 1: Speech-to-Text (Whisper via Groq) ---
    def record_voice(self) -> str | None:
        if not VOICE_AVAILABLE:
            print(f"{YELLOW}Voice not available. Run: pip install sounddevice numpy soundfile{RESET}")
            return None
        # If background listening is active, pause it so the mic is free
        was_listening = self.listen_enabled
        if was_listening:
            self.stop_listening()
            time.sleep(0.3)  # give PortAudio time to fully release the device
        print(f"{GRAY}Recording... (stops automatically when you stop talking){RESET}")
        result = self._record_until_silence()
        if was_listening:
            self.start_listening()
        return result

    # --- Feature 2: Text-to-Speech (macOS say) ---
    def speak(self, text: str):
        """Speak text via macOS say. Press any key to stop."""
        print(f"{GRAY}[speaking — press any key to stop]{RESET}")
        proc = subprocess.Popen(["say", text])
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while proc.poll() is None:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.read(1)  # consume the keypress
                    proc.kill()
                    print(f"\r{GRAY}[speech stopped]      {RESET}")
                    break
        except Exception as e:
            proc.kill()
            print(f"  {YELLOW}TTS error: {e}{RESET}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        proc.wait()

    # --- Feature 4: Sentiment Analysis ---
    def get_stats(self) -> str:
        if self.turn == 0:
            return "No conversation yet."

        user_messages = [m["content"] for m in self.history if m.get("role") == "user" and m.get("content")]
        if not user_messages:
            return "No user messages to analyze."

        analysis_prompt = (
            "Analyze these user messages. For each, give: "
            "message (truncated to 50 chars) | sentiment (positive/neutral/negative/frustrated) | emotion (1 word). "
            "Then give an overall sentiment summary in one line.\n\n"
            + "\n".join(f"{i+1}. {m[:100]}" for i, m in enumerate(user_messages))
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conversation analyst. Be concise. Use a simple table format."},
                    {"role": "user", "content": analysis_prompt},
                ],
            )
            sentiment = response.choices[0].message.content
        except Exception:
            sentiment = "Could not analyze sentiment."

        return (
            f"Turns: {self.turn}\n"
            f"Messages in context: {len(self.history)}\n"
            f"Loaded files: {len(self.loaded_files)}\n"
            f"Est. tokens: ~{self._estimate_tokens():,}\n"
            f"Speak mode: {'ON' if self.speak_enabled else 'OFF'}\n\n"
            f"--- Sentiment Analysis ---\n{sentiment}"
        )

    # --- Feature 5: Conversation Export ---
    def export_conversation(self) -> str:
        export = {
            "exported_at": datetime.datetime.now().isoformat(),
            "model": self.model,
            "turns": self.turn,
            "estimated_tokens": self._estimate_tokens(),
            "loaded_files": self.loaded_files,
            "speak_enabled": self.speak_enabled,
            "timestamps": self.timestamps,
            "messages": [
                {"role": m.get("role", ""), "content": (m.get("content") or "")[:500]}
                for m in self.history if m.get("role") in ("user", "assistant", "system")
            ],
        }
        filename = f"borisbot_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(export, f, indent=2)
        return filename

    def reset(self):
        self.turn = 0
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.timestamps = []

    # --- Feature 6: Background Listening with Wake Word ---
    def start_listening(self):
        if not VOICE_AVAILABLE:
            print(f"{YELLOW}Voice not available. Run: pip install sounddevice numpy soundfile{RESET}")
            return
        self.listen_enabled = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    def stop_listening(self):
        self.listen_enabled = False
        if VOICE_AVAILABLE:
            sd.stop()  # interrupt any blocking sd.rec()/sd.wait() in the listen thread
            time.sleep(0.3)  # give PortAudio time to fully release the device
        if self._listen_thread:
            self._listen_thread.join(timeout=6)
            self._listen_thread = None

    def _listen_loop(self):
        sample_rate = 16000
        chunk_seconds = 4
        while self.listen_enabled:
            try:
                audio = sd.rec(int(chunk_seconds * sample_rate),
                               samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()
            except Exception:
                break  # recording was interrupted (e.g. sd.stop()), exit cleanly

            if not self.listen_enabled:
                break

            # Skip silence — don't waste an API call
            if np.abs(audio).mean() < 0.005:
                continue

            try:
                # Transcribe chunk via Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp = f.name
                    sf.write(tmp, audio, sample_rate)
                with open(tmp, "rb") as af:
                    result = self.client.audio.transcriptions.create(
                        model="whisper-large-v3", file=af)
                os.unlink(tmp)

                if "hello bot" in result.text.lower():
                    print(f"\n{GREEN}Wake word detected! Listening for your command...{RESET}")
                    text = self._record_until_silence(sample_rate)
                    if text:
                        print(f"{GREEN}You said:{RESET} {text}")
                        reply = self.chat(text)
                        print(f"{GRAY}[Turn {self.turn}]{RESET} {CYAN}{BOLD}BorisBot:{RESET} {reply}\n")
                        if self.speak_enabled:
                            self.speak(reply)
                    else:
                        print(f"{YELLOW}Didn't catch that.{RESET}\n")
            except Exception:
                continue

    def _record_until_silence(self, sample_rate=16000, max_seconds=10):
        """Record audio until silence is detected, then transcribe."""
        frames = []
        silent_chunks = 0
        chunk_duration = 0.5  # 500ms per chunk
        chunk_samples = int(sample_rate * chunk_duration)
        max_silent = 3        # 1.5s of silence → stop
        min_chunks = 4        # record at least 2s before silence detection kicks in
        max_chunks = int(max_seconds / chunk_duration)

        for i in range(max_chunks):
            chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            frames.append(chunk)
            # Only start checking for silence after the minimum window
            if i >= min_chunks:
                if np.abs(chunk).mean() < 0.008:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                if silent_chunks >= max_silent:
                    break

        if not frames:
            return None

        audio = np.concatenate(frames)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
            sf.write(tmp, audio, sample_rate)
        try:
            with open(tmp, "rb") as af:
                result = self.client.audio.transcriptions.create(
                    model="whisper-large-v3", file=af)
            os.unlink(tmp)
            return result.text
        except Exception:
            os.unlink(tmp)
            return None


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
  {YELLOW}/voice{RESET}         — speak to BorisBot (speech-to-text via Whisper)
  {YELLOW}/listen{RESET}        — toggle always-on listening (say HELLO BOT to activate)
  {YELLOW}/speak{RESET}         — toggle text-to-speech on/off
  {YELLOW}/temp [0-2]{RESET}    — get/set temperature (default 0.5 — lower=direct, higher=creative)
  {YELLOW}/stats{RESET}         — conversation stats + sentiment analysis
  {YELLOW}/export{RESET}        — export conversation to JSON
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
║        BorisBot v2.0         ║
║  Voice + Agentic AI assistant║
╚══════════════════════════════╝{RESET}

{GREEN}Hey Boris! BorisBot here — now with voice, tools & analytics.
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
            bot.stop_listening()
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

        elif user_input.lower() == "/voice":
            text = bot.record_voice()
            if text:
                print(f"{GREEN}You said:{RESET} {text}")
                user_input = text
            else:
                print(f"{YELLOW}No speech detected.{RESET}\n")
                continue

        elif user_input.lower() == "/listen":
            if bot.listen_enabled:
                bot.stop_listening()
                print(f"{CYAN}BorisBot:{RESET} Background listening is now {RED}OFF{RESET}\n")
            else:
                bot.start_listening()
                if bot.listen_enabled:
                    print(f"{CYAN}BorisBot:{RESET} Background listening is now {GREEN}ON{RESET} — say {BOLD}HELLO BOT{RESET} to activate voice\n")
            continue

        elif user_input.lower() == "/speak":
            bot.speak_enabled = not bot.speak_enabled
            state = f"{GREEN}ON{RESET}" if bot.speak_enabled else f"{RED}OFF{RESET}"
            print(f"{CYAN}BorisBot:{RESET} Text-to-speech is now {state}\n")
            continue

        elif user_input.lower().startswith("/temp"):
            arg = user_input[5:].strip()
            if arg:
                try:
                    val = float(arg)
                    if 0.0 <= val <= 2.0:
                        bot.temperature = val
                        print(f"{CYAN}BorisBot:{RESET} Temperature set to {YELLOW}{val}{RESET}\n")
                    else:
                        print(f"{YELLOW}Temperature must be between 0.0 and 2.0{RESET}\n")
                except ValueError:
                    print(f"{YELLOW}Usage: /temp 0.5{RESET}\n")
            else:
                print(f"{CYAN}BorisBot:{RESET} Temperature is {YELLOW}{bot.temperature}{RESET} (0=deterministic, 2=max creative)\n")
            continue

        elif user_input.lower() == "/stats":
            print(f"{CYAN}BorisBot:{RESET}\n{bot.get_stats()}\n")
            continue

        elif user_input.lower() == "/export":
            filename = bot.export_conversation()
            print(f"{CYAN}BorisBot:{RESET} Conversation exported to {GREEN}{filename}{RESET}\n")
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
        if bot.speak_enabled:
            bot.speak(reply)


if __name__ == "__main__":
    main()
