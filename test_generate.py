import json
from pathlib import Path
import subprocess
import sys
import unittest
from generate import MarkovGenerator, Options


class GenerationTests(unittest.TestCase):
    def test_known_transition(self):
        self.assertEqual(MarkovGenerator('one two three').generate('one', Options(2)), 'two three')

    def test_unknown_word(self):
        self.assertEqual(MarkovGenerator('one two').generate('missing', Options()), '')

    def test_seed(self):
        model = MarkovGenerator('one two one three one four')
        self.assertEqual(model.generate('one', Options()), model.generate('one', Options()))

    def test_token_limit(self):
        output = MarkovGenerator('a b a b a').generate('a', Options(3))
        self.assertEqual(len(output.split()), 3)

    def test_invalid_inputs(self):
        for count in (0, 513):
            with self.assertRaises(ValueError):
                Options(count).validate()
        with self.assertRaises(ValueError):
            MarkovGenerator('one')
        with self.assertRaises(ValueError):
            MarkovGenerator('one two').generate('', Options())

    def test_cli(self):
        run = subprocess.run([sys.executable, str(Path(__file__).with_name('generate.py')),
                              '--prompt', 'Engineering', '--max-new-tokens', '10'],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(run.stdout)['backend'], 'markov')


if __name__ == '__main__':
    unittest.main()
