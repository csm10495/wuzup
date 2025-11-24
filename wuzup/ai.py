"""
Home to AI related utilities for wuzup.
"""

import logging
import os
from functools import lru_cache

import backoff
from dotenv import load_dotenv
from litellm import completion

from wuzup.exceptions import ModelNotSetError, NonBooleanAIResponseError

BOOLEAN_PROMPT_TEMPLATE = """
You are a helpful assistant that answers the question with a simple "yes" or "no".
If you are unsure you answer "no". Do not provide additional information or context around your answer.
Attempt to factually answer the question based on your training data. If necessary use knowledge, common sense, and/or math to determine the correct answer.

Here is the question:
{question}

Yes or No?
"""
YES = ("yes", "true", "1")
NO = ("no", "false", "0")

log = logging.getLogger(__name__)


def get_model() -> str:
    """
    Gets the AI model from the environment variable.
    Raises ModelNotSetError if the model is not set.
    """
    if load_dotenv():
        log.debug("Loaded environment variables from .env file")
    model = os.environ.get("WUZUP_AI_MODEL")
    if not model:
        raise ModelNotSetError(
            "WUZUP_AI_MODEL environment variable is not set. Please set it to the desired litellm model name. (For example: 'ollama_chat/deepseek-r1:1.5b')"
        )
    log.debug(f"Using AI model {model}")
    return model


@lru_cache()
@backoff.on_exception(backoff.expo, NonBooleanAIResponseError, max_tries=5, logger=log)
def ask_ai_for_boolean_response(prompt: str) -> bool:
    """
    Asks the AI a yes/no question and returns True for yes, False for no.
    """
    model = get_model()
    response = completion(
        model=model, messages=[{"role": "user", "content": BOOLEAN_PROMPT_TEMPLATE.format(question=prompt)}]
    )

    log.debug(f"Asking {model} this prompt: \n{prompt}")

    # handle if it says yes or no.. then a setence or a period.
    answer = response.choices[0].message.content.strip().lower().split()[0].split(".")[0]
    if answer in YES:
        log.debug(f"AI responded with YES to prompt: {prompt}")
        return True
    elif answer in NO:
        log.debug(f"AI responded with NO to prompt: {prompt}")
        return False
    else:
        log.error(f"Unexpected AI response for boolean question: {answer}")
        raise NonBooleanAIResponseError(f"Unexpected AI response for boolean question: {answer}")


class AISupport:
    """
    Instead of writting good logic for things, let AI do it.
    """

    def __eq__(self, other: object) -> bool:
        """
        Returns True if the two objects are considered equivalent by AI, False otherwise.
        """
        if not isinstance(other, AISupport):
            return False

        # If at the object level their vars are the same, no need to call AI.
        if vars(self) == vars(other):
            return True

        # Wish us luck
        return ask_ai_for_boolean_response(
            prompt=f"Are the two objects equivalent?: \nObject A: {self._coerce_for_ai_equivalence()}\nObject B: {other._coerce_for_ai_equivalence()}"
        )

    def _coerce_for_ai_equivalence(self) -> object:
        """
        Override this method to return a representation of the object suitable for AI equivalence checking.
        """
        return self

    def ask_ai_for_boolean_response_about_this(self, question: str) -> bool:
        """
        Ask AI a boolean question about this object.
        """
        return ask_ai_for_boolean_response(
            prompt=f"Considering this object: {self._coerce_for_ai_equivalence()}\nQuestion: {question}"
        )
