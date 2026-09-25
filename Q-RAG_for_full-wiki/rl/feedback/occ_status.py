import logging
from typing import List, Optional

from rl.feedback.feedback import AFeedbackModel
from vLLM_clients.vllm_occ import VllmOcc

logger = logging.getLogger(__name__)

OPEN_QA_TASKS = {
    "HotPotQA",
    "Musique",
    "2WikiMultihopQA",
    "HotPotQA+2WikiMultihopQA",
    "NQ+HotPotQA",
}



class OccStatus(AFeedbackModel):
    FEEDBACK_MODEL_NAME = "OCC-Status"

    def __init__(self, llm_path: str, gpu_util: float, max_tokens: int):
        super().__init__(never_terminate=True)

        self.llm_path = llm_path
        self.gpu_util = gpu_util
        self.max_tokens = max_tokens

        self.vllm_occ = VllmOcc(
            llm_path=llm_path,
            gpu_util=gpu_util,
            max_tokens=max_tokens,
        )


    def copy(self):
        return OccStatus(
            llm_path=self.llm_path,
            gpu_util=self.gpu_util,
            max_tokens=self.max_tokens
        )


    def reset(self, obs: dict | List[dict], info: dict | List[dict]) -> None:
        super().reset(obs, info)


    def reward(self, obs, info, is_final):
        if not is_final:
            return 0.0

        question = obs["question"]
        answer = str(info["answer"]).strip()
        chunks = obs.get("pred_chunks", [])
        reward = 0.0

        try:
            formatted_docs = [{"text": str(chunk)} for chunk in chunks]
            prediction, status = self.vllm_occ.chat_completion(query=question, docs=formatted_docs)
            reward = 1.0 if "ANSWERABLE" in prediction.upper() else 0.0

        except Exception as exc:
            logger.error("OccStatus reward failed: %s", exc)

        return reward
