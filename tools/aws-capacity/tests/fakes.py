"""Test doubles for the `call(args, region)` function every AWS reader takes."""


class FakeCall:
    """Records every call; answers from `responses`, a list of dicts or exceptions
    consumed in order, or from `by_op`, keyed by the AWS operation (args[1])."""

    def __init__(self, responses=None, by_op=None):
        self.responses = list(responses or [])
        self.by_op = by_op or {}
        self.calls = []

    def __call__(self, args, region=None):
        self.calls.append((list(args), region))
        if self.by_op:
            answer = self.by_op[args[1]]
            if isinstance(answer, list):
                answer = answer.pop(0)
        else:
            answer = self.responses.pop(0)
        if callable(answer) and not isinstance(answer, dict):
            answer = answer(args, region)
        if isinstance(answer, BaseException):
            raise answer
        return answer
