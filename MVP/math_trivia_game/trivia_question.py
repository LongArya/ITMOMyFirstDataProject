from dataclasses import dataclass
from typing import List


@dataclass
class TriviaQuestion:
    question_text: str
    all_answers: List[str]
    correct_answer: str

    def is_answer_correct(self, answer: str) -> bool:
        return self.correct_answer == answer
