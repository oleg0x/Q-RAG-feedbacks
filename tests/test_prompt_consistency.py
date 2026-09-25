from gig_pipeline.gig_common import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from rl.feedback.candidate_beta_feedback import (
    QA_INSTRUCTION_PROMPT,
    QA_PROMPT,
)


def test_candidate_beta_prompt_is_byte_identical_to_generation_prompt():
    question = "Does this exact question keep its trailing?"
    context = "Title One sentence.\n\nTitle Two sentence."
    assert QA_INSTRUCTION_PROMPT == QA_SYSTEM_PROMPT
    assert QA_PROMPT.format(
        context=context, question=question
    ) == QA_USER_TEMPLATE.format(context=context, question=question)
