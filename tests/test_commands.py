import unittest

from src.lib.commands import parse_destination


class ParseDestinationTests(unittest.TestCase):
    def test_direct_commands(self):
        self.assertEqual(parse_destination("start"), "start")
        self.assertEqual(parse_destination("goal"), "goal")
        self.assertEqual(parse_destination("underwater"), "underwater")
        self.assertEqual(parse_destination("go to the underwater goal"), "underwater")

    def test_natural_language_and_synonyms(self):
        self.assertEqual(parse_destination("Please return home"), "start")
        self.assertEqual(parse_destination("Go to the destination"), "goal")

    def test_unknown_or_ambiguous_command(self):
        self.assertIsNone(parse_destination("move somewhere"))
        self.assertIsNone(parse_destination("go from start to goal"))

    def test_windows_console_control_characters(self):
        self.assertEqual(parse_destination("\x00goal\xe0"), "goal")


if __name__ == "__main__":
    unittest.main()
