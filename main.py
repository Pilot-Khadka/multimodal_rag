import os
import subprocess
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from rag import RAGApp


@dataclass
class CLIConfig:
    interactive: bool = False
    verbose: bool = True
    config_file: Optional[str] = None


class CLIApp:
    def __init__(self, config: CLIConfig = CLIConfig):
        self.config = config
        self.rag = None

    def _init_rag(self):
        if self.rag is None:
            self.rag = RAGApp()

    def _print_banner(self):
        print("╭─────────────────────────────────────────╮")
        print("│         🔍 RAG CLI Application          │")
        print("│    Multimodal Retrieval & Generation    │")
        print("╰─────────────────────────────────────────╯")
        print()
        print("Type your queries below. Commands:")
        print("  /help    - Show help")
        print("  /clear   - Clear screen")
        print("  /image   - Query with image")
        print("  /exit    - Exit application")
        print()

    def _handle_command(self, user_input: str) -> tuple[bool, bool]:
        """Handle special commands. Returns (should_exit, was_command)."""
        if not user_input.startswith("/"):
            return False, False  # not a command

        parts = user_input.strip().split(" ", 2)
        command = parts[0][1:].lower()

        if command == "image":
            if len(parts) < 2:
                print("usage: /image <path_to_image> [optional_query]")
                print("xxample: /image ./photo.jpg What's in this image?")
                return False, True

            image_path = parts[1]
            # image_path = os.path.join(os.getcwd(), image_path)
            image_query = parts[2] if len(parts) > 2 else "Describe this image"
            self._handle_image(image_path, image_query)
            return False, True

        if command in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            return True, True

        elif command == "help":
            self._show_help()

        elif command == "clear":
            os.system("clear" if os.name == "posix" else "cls")
            self._print_banner()

        else:
            print(f"Unknown command: /{command}")
            print("Type /help for available commands")

        return False, True  # don't exit, but was a command

    def _handle_image(self, image_path: str, query: str):
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return

        image = Image.open(image_path)
        print(f"loaded image: {image_path} (Size: {image.size}, Mode: {image.mode})")
        print("query:", query)
        self._preview_image(image_path)

    def _preview_image(self, image_path):
        if os.name == "posix":  # macos/ linux
            subprocess.run(["xdg-open", image_path])  # or "open" on macOS
        elif os.name == "nt":  # windows
            os.startfile(image_path)

    def _show_help(self):
        print("\n📖 RAG CLI Help:")
        print("─" * 40)
        print("• Type natural language queries to search your knowledge base")
        print("• Use /commands for special functions")
        print("• Press Ctrl+C to interrupt a query")
        print("• Press Tab for command completion")
        print("\nExample queries:")
        print("  'What videos discuss rainfall simulators?'")
        print("  'Show me images related to climate change'")
        print("  '/image <path_to_image> Which video is this image from?")
        print()

    def process_query(self, query: str) -> str:
        self._init_rag()

        if self.config.verbose:
            print(f"🔍 Processing query: {query}")

        if len(query.split(" ")) < 4:
            confirm = (
                input(
                    f"⚠️ Query is very short. Are you sure you want to proceed? (y/n): "
                )
                .strip()
                .lower()
            )
            if confirm != "y":
                return "Query canceled."

        try:
            response = self.rag.query(query.strip())
            return response
        except KeyboardInterrupt:
            return "❌ Query interrupted by user"
        except Exception as e:
            error_msg = f"❌ Error processing query: {e}"
            if self.config.verbose:
                import traceback

                error_msg += f"\n{traceback.format_exc()}"
            return error_msg

    def interactive_mode(self):
        self._print_banner()

        while True:
            try:
                user_input = input("🔍 rag> ").strip()
                if not user_input:
                    continue

                should_exit, was_command = self._handle_command(user_input)
                if should_exit:
                    break

                if was_command:
                    continue

                print()
                response = self.process_query(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break

    def single_query(self, query: str):
        response = self.process_query(query)
        print(response)


if __name__ == "__main__":
    cli = CLIApp()
    cli.interactive_mode()
