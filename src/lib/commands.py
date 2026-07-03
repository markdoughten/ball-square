"""Text command parsing for interactive robot navigation."""

import re
import os
import sys


DESTINATION_WORDS = {
    "start": {"start", "beginning", "origin", "home"},
    "goal": {"goal", "target", "destination", "end"},
    "underwater": {"underwater", "water", "submerged", "fluid"},
}


def parse_destination(command):
    """Return ``start`` or ``goal`` when a destination intent is unambiguous."""
    # Console polling on Windows may interleave NUL/extended-key markers with
    # printable characters. Remove those markers before identifying words.
    normalized = "".join(
        character for character in command.lower()
        if character.isprintable() and character not in {"\x00", "\xe0"}
    )
    words = set(re.findall(r"[a-z]+", normalized))
    matches = [
        destination
        for destination, keywords in DESTINATION_WORDS.items()
        if words & keywords
    ]
    # "Underwater goal" names the specialized goal rather than two requests.
    if set(matches) == {"goal", "underwater"}:
        return "underwater"
    return matches[0] if len(matches) == 1 else None


_console_buffer = ""


def poll_console_line():
    """Return a completed console line without blocking the simulation loop."""
    global _console_buffer
    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            character = msvcrt.getwch()
            if character in {"\x00", "\xe0"}:
                # Extended keys are encoded as a prefix plus a scan code.
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if character in {"\r", "\n"}:
                print()
                line, _console_buffer = _console_buffer, ""
                return line.strip()
            if character == "\b":
                if _console_buffer:
                    _console_buffer = _console_buffer[:-1]
                    print("\b \b", end="", flush=True)
            elif character.isprintable():
                _console_buffer += character
                print(character, end="", flush=True)
        return None

    # GLFW is normally run from an interactive terminal, where select allows a
    # non-blocking stdin check on Unix-like systems.
    import select

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.readline().strip() if readable else None
