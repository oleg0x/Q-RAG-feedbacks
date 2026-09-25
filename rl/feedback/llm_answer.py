"""Q-ICL/reader feedback backed by an OpenAI-compatible vLLM server."""

import logging
from typing import List, Optional

from rouge import Rouge

from envs.dataloaders.base import separator
from prompts_and_metrics import prompts
from rl.feedback.feedback import AFeedbackModel
from utils.process_latex import extract_boxed_expression
from vLLM_clients.vllm_client import BasicVllmClient

logger = logging.getLogger(__name__)

OPEN_QA_TASKS = {
    "HotPotQA",
    "Musique",
    "2WikiMultihopQA",
    "HotPotQA+2WikiMultihopQA",
    "NQ+HotPotQA",
}


def prepare_examples(examples):
    prepared = []
    for example in examples:
        parts = example.split(separator, 1)
        if len(parts) == 2:
            prepared.append((parts[0].strip(), parts[1].strip()))
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
        api_key: Optional[str] = None,
    ):
        super().__init__(never_terminate=True)
        if task not in prompts.sys_prompts:
            raise ValueError(f"Unsupported LlmAnswer task: {task}")

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.task = task
        self.api_key = api_key
        self.rouge = Rouge()

        self.vllm_client = BasicVllmClient(
            model=model,
            base_url=base_url,
            sys_prompt=prompts.sys_prompts[task],
            api_key=api_key,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        self.vllm_client_judge = BasicVllmClient(
            model=model,
            base_url=base_url,
            sys_prompt=prompts.sys_judge,
            api_key=api_key,
            max_tokens=500,
            thinking=False,
        )

        self.last_metrics = {}


    def copy(self):
        return LlmAnswer(
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            task=self.task,
            api_key=self.api_key,
        )


    def reset(self, obs: dict | List[dict], info: dict | List[dict]) -> None:
        super().reset(obs, info)
        self.last_metrics = {}


    def reward(self, obs, info, is_final):
        if not is_final:
            return 0.0

        question = obs["question"]
        answer = str(info["answer"]).strip()
        chunks = obs.get("pred_chunks", [])
        examples = None
        passages = chunks

        if self.task not in OPEN_QA_TASKS:
            examples = prepare_examples(chunks)
            passages = None

        prediction = ""
        judgment = ""
        reward = 0.0
        em = 0
        f1 = 0.0

        try:
            response, mean_logprob = self.vllm_client.chat_completion(
                query=question,
                examples=examples,
                passages=passages,
            )
            prediction = discard_reasoning(response).strip()

            if self.task in OPEN_QA_TASKS or self.task == "GSM8K":
                prediction = get_final_answer(prediction)
            elif self.task == "MATH":
                prediction = extract_boxed_expression(prediction)

            normalized_prediction = prediction.strip().lower()
            normalized_answer = answer.lower()

            em = int(normalized_prediction == normalized_answer)
            if self.task == "XL-Sum" and normalized_prediction:
                scores = self.rouge.get_scores(normalized_prediction, normalized_answer)
                f1 = float(scores[0]["rouge-l"]["f"])
                reward = f1
            elif em:
                reward = 1.0
            elif normalized_prediction and normalized_answer:
                judgment, _ = self.vllm_client_judge.chat_completion(
                    query=question,
                    two_answers=(normalized_prediction, normalized_answer),
                )
                judgment = get_final_answer(judgment).upper()
                reward = float(judgment == "CORRECT")

        except Exception as exc:
            mean_logprob = None
            logger.error("LlmAnswer reward failed: %s", exc)

        # self.last_metrics = {
        #     "pred": prediction,
        #     "EM": em,
        #     "F1": f1,
        #     "LLM-as-judge": reward,
        #     "mean_logprob": mean_logprob,
        #     "judgment": judgment,
        # }
        return reward


    def get_metrics(self):
        return dict(self.last_metrics)
