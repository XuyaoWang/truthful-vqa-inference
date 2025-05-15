import os
import copy
from typing import Dict, Any, Tuple
from datasets import load_dataset, concatenate_datasets

from truthful_vqa_inference.client.evaluation import BenchmarkEvaluator
from truthful_vqa_inference.benchmarks.registry import BenchmarkRegistry

@BenchmarkRegistry.register("truthfulvqa")
class TruthfulVQAEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the TruthfulVQA benchmark.
    """
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the TruthfulVQA dataset and duplicate it with different categories.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")
            
        Returns:
            The loaded dataset with duplicated entries for multi-choice and open-qa
        """
        
        dataset = load_dataset(self.data_path)['validation']

        num_processors = min(os.cpu_count(), self.num_workers)
        
        # Create deep copies of the dataset with different categories
        multi_choice_dataset = dataset.map(
            lambda x: {**copy.deepcopy(x), "task-type": "multi-choice"},
            num_proc=num_processors
        )
        open_qa_dataset = dataset.map(
            lambda x: {**copy.deepcopy(x), "task-type": "open-qa"},
            num_proc=num_processors
        )
        
        # Concatenate the two datasets
        combined_dataset = concatenate_datasets([multi_choice_dataset, open_qa_dataset])
        
        return combined_dataset
    
    
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a single item from the dataset.
        
        Args:
            item: A single item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        question = item['question']
        options = item['options']
        image = item['image']

        system_content = ""
        user_content = question + "\n\n"

        if item['task-type'] == 'multi-choice':
            for idx, option in enumerate(options):
                user_content += f"({chr(ord('A') + idx)}) {option}\n"
            user_content += "\n\nAnswer with the option's letter from the given choices directly."
        else:
            user_content += "\n\nAnswer the question using a single word or phrase."

        return system_content, user_content, image
    
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a single model output against an item from the dataset.
        
        Args:
            item: A single item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        # Return detailed results
        return {
            "question_id": item['question_id'],
            "question": item['question'],
            "task-type": item['task-type'],
            "category": item['category'],
            "subcategory": item['subcategory'],
            "level": item['level'],
            "answer": item['answer'],
            "response": response,
        }