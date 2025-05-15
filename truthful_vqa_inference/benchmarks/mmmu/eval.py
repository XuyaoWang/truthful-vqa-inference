from typing import Dict, Any, List, Tuple, Optional

import re
import os
from datasets import load_dataset, concatenate_datasets

from truthful_vqa_inference.client.evaluation import BenchmarkEvaluator
from truthful_vqa_inference.benchmarks.registry import BenchmarkRegistry

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

@BenchmarkRegistry.register("mmmu")
class MMMUEvaluator(BenchmarkEvaluator):
    """Evaluator for the MMMU benchmark."""
    
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the MMMU dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")
            
        Returns:
            The loaded dataset
        """
        categories = list(CAT_SHORT2LONG.values())

        num_processors = min(os.cpu_count(), self.num_workers)
        return concatenate_datasets([
            (lambda d: d.add_column('category', [category] * len(d)))(
                load_dataset(self.data_path, category, split='validation', num_proc=num_processors)
            )
            for category in categories
        ])
    
    def get_image_indice(self, text: str)->List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]
    
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for an item from the dataset.
        
        Args:
            item: An item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """

        system_content = ""
        question = item['question']
        if item['question_type'] == 'multiple-choice':
            options = eval(item['options'])
            example = ""
            letter_to_option = {}
                
            for idx, option in enumerate(options):
                option_letter = chr(ord('A') + idx)
                example += f"({option_letter}) {option}\n"
                letter_to_option[option_letter] = option
                
            user_content = f"{question}\n\n{example}\n\nAnswer with the option's letter from the given choices directly."
        else:
            user_content = f"{question}\n\nAnswer the question using a single word or phrase."
            
        image_ids = self.get_image_indice(user_content)
        image = [item[f'image_{id}'] for id in image_ids]

        return system_content, user_content, image
    
    def prepare_messages(self, system_content: str, user_content: str, image: Any) -> List[Dict[str, Any]]:
        """
        Prepare messages for the API.
        
        Args:
            system_content: System prompt content
            user_content: User prompt content
            image: Image or list of images

        Returns:
            List of messages ready for the API
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_content))
        images = image if isinstance(image, list) else [image]
        
        if matches:
            assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_content}"
            
            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append({"type": "text", "text": user_content[last_end:match.start()]})
                content_parts.append({"type": "image_url", "image_url": {"url": self.encode_image(images[i])}})
                last_end = match.end()
                
            if last_end < len(user_content):
                content_parts.append({"type": "text", "text": user_content[last_end:]})
        else:
            content_parts.extend([{"type": "image_url", "image_url": {"url": self.encode_image(img)}} for img in images])
            if user_content:
                content_parts.append({"type": "text", "text": user_content})

        messages = [{"role": "user", "content": content_parts}]
        if system_content:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_content}]})

        return messages
    
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model output against an item from the dataset.
        
        Args:
            item: An item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        return {
            "id": item['id'],
            "category": item['category'],
            "question": item['question'],
            "response": response,
            "answer": item['answer'],
        }