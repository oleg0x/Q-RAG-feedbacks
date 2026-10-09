import logging
from typing import List, Optional

from envs.dataloaders.base import separator
from prompts_and_metrics import prompts
from rl.feedback.feedback import AFeedbackModel
from vLLM_clients.vllm_client import BasicVllmClient

logger = logging.getLogger(__name__)

OPEN_QA_TASKS = {
    "HotPotQA",
    "Musique",
    "2WikiMultihopQA",
    "HotPotQA+2WikiMultihopQA",
    "NQ+HotPotQA",
}
_OPEN_QA_TASKS_LOWER = {t.lower() for t in OPEN_QA_TASKS}


def prepare_examples(examples):
    prepared = []
    for ex in examples:
        split_parts = ex.split(separator, 1)
        if len(split_parts) == 2:
            problem = split_parts[0].strip()
            solution = split_parts[1].strip()
            prepared.append((problem, solution))
    return prepared


def discard_reasoning(response: str) -> str:
    if "<think>" not in response:
        return response
    if "</think>" not in response:
        return "[incomplete reasoning]"
    return response.rsplit("</think>", 1)[1].strip()


def get_final_answer(text: str) -> str:
    marker = "final answer:"
    position = text.lower().rfind(marker)
    if position != -1:
        text = text[position + len(marker) :]
    return text.strip().rstrip(".")


class LlmAnswer(AFeedbackModel):
    FEEDBACK_MODEL_NAME = "LLM-Answer"

    def __init__(self,
        model: str,
        base_url: str,
        max_tokens: int,
        thinking: bool,
        task: str,
        reward_type: str,
        api_key: Optional[str] = None,
    ):
        super().__init__(never_terminate=True)

        if task.lower() not in _OPEN_QA_TASKS_LOWER:
            raise ValueError(f"Unsupported LlmAnswer task: {task}")
        if reward_type not in {'exact-match', 'llm-judge'}:
            raise ValueError(f"Unsupported LlmAnswer reward_type: {reward_type}")

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.task = task
        self.api_key = api_key
        self.reward_type = reward_type

        self.vllm_client = BasicVllmClient(
            model=model,
            base_url=base_url,
            sys_prompt=prompts.sys_prompts[task],
            api_key=api_key,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        if self.reward_type == 'llm-judge':
            self.vllm_client_judge = BasicVllmClient(
                model=model,
                base_url=base_url,
                sys_prompt=prompts.sys_judge,
                api_key=api_key,
                max_tokens=500,
                thinking=False,
            )


    def copy(self):
        return LlmAnswer(
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            task=self.task,
            reward_type=self.reward_type,
            api_key=self.api_key,
        )


    def reset(self, obs: dict | List[dict], info: dict | List[dict]) -> None:
        super().reset(obs, info)


    def reward(self, obs, info, is_final):
        if not is_final:
            return 0.0

        question = obs["question"]
        answer = str(info["answer"]).strip()
        chunks = obs.get("pred_chunks", [])

        prediction = ""
        judgment = ""
        reward = 0.0
        em = 0

        try:
            response, _ = self.vllm_client.chat_completion(
                query=question,
                passages=chunks,
            )
            prediction = discard_reasoning(response).strip()
            prediction = get_final_answer(prediction)

            normalized_prediction = prediction.strip().lower()
            normalized_answer = answer.lower()

            em = int(normalized_prediction == normalized_answer)
            reward = float(em)

            if self.reward_type == 'llm-judge' and em == 0:
                judgment, _ = self.vllm_client_judge.chat_completion(
                    query=question,
                    two_answers=(normalized_prediction, normalized_answer),
                )
                judgment = discard_reasoning(judgment)
                judgment = get_final_answer(judgment).upper()
                reward = float(judgment == "CORRECT")

        except Exception as exc:
            logger.error("LlmAnswer reward failed: %s", exc)

        return reward
