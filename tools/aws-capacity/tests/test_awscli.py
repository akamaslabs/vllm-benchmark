import subprocess
import unittest
from unittest import mock

import awscli


class ParseErrorTest(unittest.TestCase):
    def test_parse_error_with_code(self):
        e = awscli.parse_error(
            "\nAn error occurred (InsufficientInstanceCapacity) when calling the "
            "CreateCapacityReservation operation: Insufficient capacity.\n")
        self.assertEqual(e.code, "InsufficientInstanceCapacity")
        self.assertIn("Insufficient capacity", e.message)

    def test_parse_error_without_code(self):
        e = awscli.parse_error(
            '\nUnable to locate credentials. You can configure credentials by running "aws configure".\n')
        self.assertEqual(e.code, "CliError")
        self.assertIn("Unable to locate credentials", e.message)


class RunTest(unittest.TestCase):
    def run_aws(self, **ret):
        done = subprocess.CompletedProcess([], ret.get("code", 0), stdout=ret.get("out", "{}"), stderr="")
        with mock.patch("awscli.subprocess.run", return_value=done) as run:
            awscli.aws(["ec2", "describe-regions"], "us-east-2")
        return run.call_args

    def test_profile_on_command_line(self):
        argv = self.run_aws().args[0]
        self.assertEqual(argv[argv.index("--profile") + 1], "lab")

    def test_child_in_own_session_without_stdin_and_with_timeout(self):
        kw = self.run_aws().kwargs
        self.assertTrue(kw["start_new_session"])
        self.assertIs(kw["stdin"], subprocess.DEVNULL)
        self.assertGreater(kw["timeout"], 0)

    def test_timeout_becomes_aws_error(self):
        with mock.patch("awscli.subprocess.run", side_effect=subprocess.TimeoutExpired(["aws"], 120)):
            with self.assertRaises(awscli.AwsError) as cm:
                awscli.aws(["ec2", "describe-regions"])
        self.assertEqual(cm.exception.code, "Timeout")


if __name__ == "__main__":
    unittest.main()
