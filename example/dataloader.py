from transformers import PreTrainedTokenizerBase
from .dataloader_extras import (
    dataset_to_system_prompt,
    dataset_to_context_prefix,
    dataset_to_prompt_prefix,
    dataset_to_prompt_suffix
    )
from datasets import load_dataset
import os

num_proc = os.cpu_count()

class TextSection:
    def __init__(self, name, text, priority, tokenizer):
        self.name = name
        self.text = text
        self.priority = priority
        self.tokenizer = tokenizer
        self.truncated = 0
        self.truncated_before = False
        self.tokenized_text = self.tokenizer(self.text, truncation=False)["input_ids"]
        self.initial_token_count = len(self.tokenized_text)

    def token_count(self):
        return len(self.tokenized_text)

    def truncate(self, num_tokens):
        if self.truncated_before:
            raise ValueError("[!] section already truncated!")
        if num_tokens >= len(self.tokenized_text):
            raise ValueError(f"[!] this should not happen for text section '{self.name}' with priority {self.priority}!")
        self.truncated_before = True
        self.tokenizer.truncation_side = 'left'
        input_ids = self.tokenizer.encode(self.text, truncation=False, add_special_tokens=False)
        input_ids = input_ids[num_tokens:]
        self.tokenized_text = input_ids
        
        truncated_text = self.tokenizer.decode(self.tokenized_text, skip_special_tokens=True)
        if len(truncated_text) < len(self.text):
            self.truncated = len(self.tokenized_text) - len(truncated_text)
        if truncated_text != self.text:
            truncated_text = " ..." + truncated_text
        self.text = truncated_text
        return self.truncated
        
    def __repr__(self):
        truncated_percentage = (self.truncated / len(self.text)) * 100 if len(self.text) > 0 else 0
        return f"<Section {self.name}:{self.priority} ({'Truncated' if self.truncated_before else 'Complete'} — {self.truncated}, {truncated_percentage:.2f}% truncated)>"


class ModelInputPreprocessor:
    def __init__(self, config: dict, silent: bool = False):
        self.dataset_to_system_prompt = dataset_to_system_prompt
        # must pass joined_retrieved_ids
        self.dataset_to_context_prefix = dataset_to_context_prefix
        self.dataset_to_prompt_prefix = dataset_to_prompt_prefix
        self.dataset_to_prompt_suffix = dataset_to_prompt_suffix
        required_keys = ['dataset', 'dataset_ratio', 'setup', 'split', 'top_k_passages']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"[!] Missing required config key: {key}")

        self.dataset = config['dataset']
        self.dataset_ratio = config['dataset_ratio']
        self.setup = config['setup']
        self.split = config['split']
        self.top_k_passages = config['top_k_passages']
        self.use_instructions = config['use_instructions'] or False

        self.current_dataset = None
        workshop_hf_name = "ylkhayat/{dataset_name}-generation-workshop"
        if self.dataset == "clerc":
            if 'noisy' in self.setup:
                raise ValueError("[!] noisy data not supported for CLERC.")
            dataset_repo_name_prefix = "CLERC"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            self.current_dataset = load_dataset(workshop_hf_name, data_dir=self.setup, split=self.split)
        elif self.dataset == "echr":
            dataset_repo_name_prefix = "ECHR"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            self.current_dataset = load_dataset(workshop_hf_name, data_dir=self.setup, split=self.split)
        elif self.dataset == "cuad":
            if 'oracle_passages' in self.setup:
                raise ValueError("[!] oracle passages data not supported for CUAD.")
            dataset_repo_name_prefix = "CUAD"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            self.current_dataset = load_dataset(workshop_hf_name, data_dir=self.setup, split=self.split)
        elif self.dataset == "obli_qa":
            if 'oracle_passages' in self.setup:
                raise ValueError("[!] oracle passages data not supported for ObliQA.")
            dataset_repo_name_prefix = "OBLI_QA"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            self.current_dataset = load_dataset(workshop_hf_name, data_dir=self.setup, split=self.split)
        elif self.dataset == "oal_qa":
            if 'oracle_passages' in self.setup:
                raise ValueError("[!] oracle passages data not supported for OAL QA.")
            dataset_repo_name_prefix = "OAL_QA"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            self.current_dataset = load_dataset(workshop_hf_name, data_dir=self.setup, split=self.split)
        elif self.dataset == "echr_qa":
            dataset_repo_name_prefix = "ECHR_QA"
            workshop_hf_name = workshop_hf_name.format(dataset_name=dataset_repo_name_prefix)
            url=f"https://huggingface.co/datasets/{workshop_hf_name}/resolve/main/{self.setup}/"
            self.current_dataset = load_dataset("parquet", data_files={self.split: f"{url}{self.split}*.parquet"})[self.split]
        assert self.current_dataset is not None, f"Dataset '{self.dataset}' not supported."
        length_of_dataset = int(len(self.current_dataset) * self.dataset_ratio)
        if not silent:
            print(f"[!] dataset: '{workshop_hf_name}'")
            print(f"[!] experiment num of records: {length_of_dataset}")

        self.processed_dataset = self.current_dataset.select(range(length_of_dataset))

    def build_context_prompt(self,
                             prompt,
                             contexts,
                             max_tokens,
                             use_instructions):
        assert (
            self.dataset in self.dataset_to_context_prefix and 
            self.dataset in self.dataset_to_prompt_prefix and 
            self.dataset in self.dataset_to_prompt_suffix and
            self.dataset in self.dataset_to_system_prompt
        ), f"Dataset '{self.dataset}' not supported."
        assert isinstance(contexts, list) and len(contexts) > 0, "Contexts must be a non-empty list of strings."
        retrieved_ids = [doc.split('\n')[0] for doc in contexts]
        def get_copy_prefix_linker():
            return TextSection("prefix_linker", "\n\n", priority=0, tokenizer=self.tokenizer)
        prefix_linker_section = get_copy_prefix_linker()
        context_prefix = self.dataset_to_context_prefix[self.dataset].format(joined_retrieved_ids=', '.join(retrieved_ids), single_retrieved_id=retrieved_ids[0])
        context_prefix_section = TextSection("context_prefix", context_prefix, priority=0, tokenizer=self.tokenizer)
        context_sections = []
        for i, context in enumerate(contexts):
            context_sections.append(TextSection(f"ref_text_{i}", context, priority=1, tokenizer=self.tokenizer))
            if i < len(contexts) - 1:
                context_sections.append(TextSection("prefix_linker", "\n\n", priority=0, tokenizer=self.tokenizer))
        prompt_prefix = self.dataset_to_prompt_prefix[self.dataset]
        
        prompt_prefix_section = TextSection("prompt_prefix", prompt_prefix, priority=0, tokenizer=self.tokenizer)
        prompt_suffix = self.dataset_to_prompt_suffix[self.dataset]
        prompt_suffix_section = TextSection("prompt_suffix", prompt_suffix, priority=0, tokenizer=self.tokenizer)
        prompt_section = TextSection("prompt", prompt, priority=1, tokenizer=self.tokenizer)
        building_sections = []
        building_sections.append([context_prefix_section, get_copy_prefix_linker()])
        building_sections.append(context_sections)
        building_sections.append([prompt_prefix_section, get_copy_prefix_linker(), prompt_section, prompt_suffix_section])
        
        all_sections = [section for group in building_sections for section in group]
        all_sections_token_count = [section.token_count() for section in all_sections]
        current_length = sum(all_sections_token_count)
        truncation = False
        if current_length > max_tokens:
            truncation = True
            extra_length = current_length - max_tokens
            minimum_tokens_per_section = 10
            weights = [(max(section.token_count() - minimum_tokens_per_section, 0) * section.priority / current_length)  if section.priority != 0 else 0 for section in all_sections]
            tokens_to_truncate = [int(extra_length * ratio) for ratio in weights]
            total_truncated = sum(tokens_to_truncate)
            remaining_truncations = extra_length - total_truncated
            if remaining_truncations > 0:
                weights = [(max(section.token_count() - minimum_tokens_per_section, 0) / current_length)  if section.priority == 0 else 0 for section in all_sections]
                tokens_to_truncate_last = [int(remaining_truncations * ratio) for ratio in weights]
                tokens_to_truncate = [truncate_sign_1 + truncate_sign_2 for truncate_sign_1, truncate_sign_2 in zip(tokens_to_truncate, tokens_to_truncate_last)]
        
            for section, truncate in zip(all_sections, tokens_to_truncate):
                if truncate > 0:
                    section.truncate(truncate)

        context_prefix = context_prefix_section.text
        ref_text = ''.join([section.text for section in context_sections])
        formatted_prompt = prompt_prefix_section.text + prefix_linker_section.text + prompt_section.text + prompt_suffix_section.text
        
        if use_instructions and 'apply_chat_template' in dir(self.tokenizer):
            formatted_prompt = self._apply_chat_template(formatted_prompt)
            
        return context_prefix, ref_text, formatted_prompt, truncation

    def _apply_chat_template(self, prompt):
        use_system = True
        if "Saul" in self.tokenizer.name_or_path:
            use_system = False
        prompt_chat_parts = []
        if use_system:
            prompt_chat_parts.append({"role": "system", "content": self.dataset_to_system_prompt[self.dataset]})
        prompt_chat_parts.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(prompt_chat_parts, tokenize=False)
    
    def preprocess_record(self, record, top_k, max_tokens, use_instructions):
        prev_text = record['previous_text']
        gold_text = record['gold_text']
        oracle_documents = record['citations']
        if 'top_10_passages' in record.keys():
            retrieved_docs = record['top_10_passages'][:top_k]
        elif 'top_k_passages' in record.keys():
            retrieved_docs = record['top_k_passages'][:top_k]
        else:
            raise ValueError("[!] no top_k_passages found in record.")

        context_prefix, contexts, prompt, is_truncated = self.build_context_prompt(prompt=prev_text,
                                                                     contexts=retrieved_docs,
                                                                     use_instructions=use_instructions,
                                                                     max_tokens=max_tokens)
        assert isinstance(is_truncated, bool), "is_truncated must be a boolean."
        return {
            "prompt": prompt,
            "context": contexts,
            "context_prefix": context_prefix,
            "is_truncated": is_truncated,
            "meta": {
                "gold_text": gold_text,
                "previous_text": prev_text,
                "oracle_documents": oracle_documents,
                "top_k_passages": retrieved_docs
            }
        }

    def process_dataset(self, tokenizer: PreTrainedTokenizerBase, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.processed_dataset = self.processed_dataset.map(
            lambda record: self.preprocess_record(record,
                                                  top_k=self.top_k_passages,
                                                  max_tokens=self.max_tokens,
                                                  use_instructions=self.use_instructions),
            batched=False,
            num_proc=num_proc
        )
        self.processed_dataset = self.processed_dataset.flatten()
        assert "meta.oracle_documents" in self.processed_dataset.column_names
        try:
            self.processed_dataset = self.processed_dataset.rename_column("appno", "docid")
        except:
            pass
        return self.processed_dataset, self.current_dataset
    
    

    