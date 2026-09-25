from torch.utils.data import Dataset
import envs.chunker
from envs.dataloaders.gsm8k import extract_short_answer
from envs.dataloaders.hellaswag import format_answer as format_hellaswag_answer, format_query as format_hellaswag_query
from envs.dataloaders.mmlupro import format_query as format_mmlupro_query
from envs.utils import wiki_style_chunk



class QADatasetAdapter(Dataset):
    """
    Simple adapter that adapts datasets Babilong, HotPotQA and MUSIQUE for QARetrievalEnv.
    This adapter doesn't tokenize or embeds text chunks.

    You can create different adapter that for example tokenize every text in a sample or
    build faiss index over text chunks.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset.name()
        #print(f"{self.dataset_name} dataset length: {self.dataset.__len__()}")


    def __getitem__(self, index):
        sample = self.dataset[index]
        sf_idx = []
        chunk_texts = []
        answer_variants = None

        if self.dataset_name == "combined":
            source = sample.get('source')
            if source not in ('hotpotqa', 'musique', 'babilong', '2WikiMultihopQA'):
                raise ValueError(f"Invalid or missing 'source' in combined dataset sample: {source}")
        else:
            source = self.dataset_name

        if source in {'hotpotqa', '2WikiMultihopQA'}:
            sample_id = sample['_id']
            question = sample["question"]
            answer = sample["answer"]
            sp_title_set = set()
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])
            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                chunk = wiki_style_chunk(title, " ".join(sentences))
                chunk_texts.append(chunk)

        elif source == 'musique':
            sample_id = sample['id']
            question = sample["question"]
            answer = sample["answer"]
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                chunk = para['title'] + '. ' + para['paragraph_text']
                chunk_texts.append(chunk)
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])

        elif source == 'babilong':
            sample_id = index
            question = sample["question"]
            answer = sample["answer"]
            chunk_texts = sample['chunks']
            sf_idx = list(sample['references_idx'])

        elif source == 'LongBench':
            sample_id = sample["_id"]
            question = sample["input"]
            answer = sample["answers"][0]
            chunk_texts = envs.chunker.chunks_split(sample["context"], chunk_size=2000)
            sf_idx.append(0)  #sf_idx = None
            #print("Chunks count:", len(chunks_texts))

        elif source == 'RulerQA':
            sample_id = sample['index']
            question = sample["question"]
            answer = sample['outputs'][0]
            chunk_texts = sample['documents']
            sf_idx.append(0)  #sf_idx = None

        elif source == "gsm8k":
            sample_id = sample.get("id", index)
            question = sample["question"]
            answer = extract_short_answer(sample["answer"])
            chunk_texts = self.dataset.get_examples()
            sf_idx = [0]

        elif source == "MATH":
            sample_id = sample["id"]
            question = sample["problem"]
            answer = sample["answer"]
            chunk_texts = self.dataset.get_examples()
            sf_idx = [0]

        elif source == "HellaSwag":
            sample_id = sample["ind"]
            question = format_hellaswag_query(sample)
            answer = format_hellaswag_answer(sample)
            chunk_texts = self.dataset.get_examples()
            sf_idx = [0]

        elif source == "XL-Sum":
            sample_id = sample["id"]
            question = sample["text"]
            answer = sample["summary"]
            chunk_texts = self.dataset.get_examples()
            sf_idx = [0]

        elif source == "MMLU-Pro":
            sample_id = sample["question_id"]
            question = format_mmlupro_query(sample)
            answer = sample["answer"]
            chunk_texts = self.dataset.get_examples()
            sf_idx = [0]

        elif source == "NQ+HotPotQA":
            sample_id = sample["id"]
            question = sample["question"]
            # golden_answers — список допустимых ответов. Основным считается
            # первый, остальные едут отдельным полем: сравнивать предсказание
            # со склейкой всех вариантов нельзя, EM не сработает ни на одном.
            answer_variants = list(sample["golden_answers"])
            answer = answer_variants[0]
            chunk_texts = ["111", "222", "333"]
            sf_idx = [0]

        else:
            raise ValueError(f"Unsupported dataset/source: {source}")

        result = {
            'id': sample_id,
            'question': question,
            'answer': answer,
            'chunks': chunk_texts,
            'sf_idx': sf_idx,
            'source': source,
        }
        if answer_variants is not None:
            result['answer_variants'] = answer_variants
        return result


    def __len__(self):
        return self.dataset.__len__()
