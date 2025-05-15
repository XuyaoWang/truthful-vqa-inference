import os
import json
import time
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from truthful_vqa_inference.utils.parallel_processing import parallel_processing_backend
from truthful_vqa_inference.utils.cached_requests import cached_requests
# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""
    
    def __init__(self, 
                 benchmark_name: str,
                 model_name: str, 
                 api_base: str,
                 data_path: Optional[str] = None,
                 api_key: str = "EMPTY",
                 cache_dir: str = "./cache",
                 results_dir: str = "./results",
                 num_workers: int = 100):
        """
        Initialize a benchmark evaluator.
        
        Args:
            benchmark_name: Specific name of the benchmark (passed from registry)
            model_name: Name of the model being evaluated
            api_base: Base URL for the API
            data_path: Optional path to the benchmark dataset
            api_key: API key for the API
            cache_dir: Directory to cache results
            results_dir: Directory to save evaluation results
            num_workers: Number of parallel workers for processing
        """
        self._benchmark_name = benchmark_name
        self.model_name = model_name
        self.api_base = api_base
        self.data_path = data_path
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.results_dir_base = os.path.join(results_dir, self._benchmark_name, model_name) 

        self.num_workers = num_workers


    @property
    def benchmark_name(self) -> str:
        """Returns the specific name of the benchmark."""
        return self._benchmark_name

    @abstractmethod
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the benchmark dataset.
        
        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")
            
        Returns:
            The loaded dataset
        """
        pass
    
    @abstractmethod
    def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Prepare input for a single item from the dataset.
        
        Args:
            item: A single item from the dataset
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        pass
    
    @abstractmethod
    def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a single model output against an item from the dataset.
        
        Args:
            item: A single item from the dataset
            response: Model's response for this item
            
        Returns:
            Evaluation results for this item
        """
        pass
    
    @staticmethod
    def encode_image(image: Union[str, Image.Image]) -> str:
        """
        Encode an image as a base64 data URL.
        
        Args:
            image: Path to an image or a PIL Image object
            
        Returns:
            Base64 encoded image as a data URL
        """
        if isinstance(image, str):
            image_input = Image.open(image)
        else:
            image_input = image
        
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
    def prepare_messages(self, system_content: str, user_content: str, image: Any) -> List[Dict[str, Any]]:
        """
        Prepare messages from the tuple of (system_content, user_content, image).
        This is the default implementation that can be overridden by derived classes.
        
        Args:
            system_content: System prompt content
            user_content: User prompt content
            image: Image or list of images
            
        Returns:
            List of message dictionaries ready for the API
        """
        # Process images
        images = []
        if image is not None:
            if isinstance(image, list):
                images = [self.encode_image(img) for img in image]
            else:
                images = [self.encode_image(image)]
        
        # Construct the content array for the user message
        content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
        content.append({"type": "text", "text": user_content})
        
        # Build the messages list
        messages = [{'role': 'user', 'content': content}]
        if system_content:
            messages.insert(0, {'role': 'system', 'content': system_content})
            
        return messages
    
    def _prepare_input_item_wrapper(self, param: Dict[str, Any]) -> Tuple[str, str, Any]:
        """
        Wrapper function for prepare_input_item to use with parallel processing.
        
        Args:
            param: Dictionary containing the item
            
        Returns:
            Tuple of (system_content, user_content, image)
        """
        return self.prepare_input_item(param['item'])
    
    def prepare_inputs(self, dataset: Any, num_workers: int = 100) -> Tuple[List[str], List[str], List[Any]]:
        """
        Prepare inputs for the budget forcing client using parallel processing.
        
        Args:
            dataset: The loaded dataset
            num_workers: Number of parallel workers
            
        Returns:
            Tuple of (system_contents, user_contents, images)
        """
        # Prepare parameter list for parallel processing
        params = [{'item': item} for item in dataset]
        
        # Run parallel processing
        logger.info(f"Preparing inputs for {len(params)} examples using {num_workers} workers")
        results = parallel_processing_backend(
            params=params,
            fn=self._prepare_input_item_wrapper,
            num_workers=num_workers,
            desc="Preparing Inputs"
        )
        
        # Unpack results
        system_contents = []
        user_contents = []
        images = []
        
        for system_content, user_content, image in results:
            system_contents.append(system_content)
            user_contents.append(user_content)
            images.append(image)
            
        return system_contents, user_contents, images
    
    def _prepare_messages_wrapper(self, param: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wrapper function for prepare_messages to use with parallel processing.
        
        Args:
            param: Dictionary containing system_content, user_content, and image
            
        Returns:
            List of message dictionaries
        """
        return self.prepare_messages(
            param['system_content'], 
            param['user_content'], 
            param['image']
        )
    
    def prepare_all_messages(self, 
                            system_contents: List[str], 
                            user_contents: List[str], 
                            images: List[Any],
                            num_workers: int = 100) -> List[List[Dict[str, Any]]]:
        """
        Prepare all messages for the budget forcing client using parallel processing.
        
        Args:
            system_contents: List of system prompts
            user_contents: List of user prompts
            images: List of images
            num_workers: Number of parallel workers
            
        Returns:
            List of message lists ready for the API
        """
        # Prepare parameter list for parallel processing
        params = [
            {
                'system_content': system_content,
                'user_content': user_content,
                'image': image
            } 
            for system_content, user_content, image in zip(system_contents, user_contents, images)
        ]
        
        # Run parallel processing
        logger.info(f"Preparing messages for {len(params)} examples using {num_workers} workers")
        results = parallel_processing_backend(
            params=params,
            fn=self._prepare_messages_wrapper,
            num_workers=num_workers,
            desc="Preparing Messages"
        )
            
        return results
    
    def _evaluate_item_wrapper(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper function for evaluate_item to use with parallel processing.
        
        Args:
            param: Dictionary containing the item and response
            
        Returns:
            Evaluation results for this item
        """
        return self.evaluate_item(param['item'], param['response'])
    
    def evaluate_outputs(self, 
                         dataset: Any, 
                         responses: List[str], 
                         num_workers: int = 100) -> Dict[str, Any]:
        """
        Evaluate model outputs against the benchmark using parallel processing.
        
        Args:
            dataset: The loaded dataset
            responses: Model responses from budget forcing
            num_workers: Number of parallel workers
            
        Returns:
            Dictionary with evaluation results
        """
        num_total = len(responses)
        
        # Prepare parameter list for parallel processing
        params = [{'item': item, 'response': response} for item, response in zip(dataset, responses)]
        
        # Run parallel processing
        logger.info(f"Evaluating {len(params)} examples using {num_workers} workers")
        detailed_results = parallel_processing_backend(
            params=params,
            fn=self._evaluate_item_wrapper,
            num_workers=num_workers,
            desc="Evaluating Outputs"
        )
        
        # Calculate metrics using the potentially overridden method
        return self.calculate_metrics(detailed_results)
    
    def inference(self,
                  messages_list: List[List[Dict[str, Any]]],
                  temperature: float,
                  top_p: float,
                  repetition_penalty: float,
                  max_tokens: int,
                  ) -> List[Any]:
        """
        Run inference on the model using parallel processing.
        
        Args:
            messages_list: List of message lists
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens: Maximum tokens for the response
            
        Returns:
            List of responses
        """
        params = [{
            'messages': messages, 
            'temperature': temperature, 
            'top_p': top_p, 
            'repetition_penalty': repetition_penalty, 
            'max_tokens': max_tokens,
            } for messages in messages_list]
        
        results = parallel_processing_backend(
            params=params,
            fn=self._inference_item_wrapper,
            num_workers=self.num_workers,
            desc="Inferring"
        )
        return results

    def _inference_item_wrapper(self, param: Dict[str, Any]) -> Any:
        """Wrapper for inference_item for parallel processing."""
        return self.inference_item(
            messages=param["messages"],
            temperature=param["temperature"],
            top_p=param["top_p"],
            repetition_penalty=param["repetition_penalty"],
            max_tokens=param["max_tokens"]
        )

    def inference_item(self,
                       messages: List[Dict[str, Any]],
                       temperature: float,
                       top_p: float,
                       repetition_penalty: float,
                       max_tokens: int
                       ) -> Any:
        """Calls the cached request function for a single inference."""
        response = cached_requests(
            messages=messages,
            model=self.model_name,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_completion_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            cache_dir=self.cache_dir
        )
        return response
    
    def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on detailed evaluation results.
        
        This method can be overridden by derived classes to implement custom metrics.
        
        Args:
            detailed_results: List of detailed evaluation results for each item
            
        Returns:
            Dictionary with evaluation metrics
        """
        num_total = len(detailed_results)

        # Count correct answers
        num_match = sum(1 for result in detailed_results if result.get('correct', False))
        accuracy = num_match / num_total if num_total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "num_correct": num_match,
            "num_total": num_total,
            "detailed_results": detailed_results
        }
    
    def run_evaluation(
        self,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        max_tokens: int = 2048,
        split: str = "test"
    ) -> Dict[str, Any]:
        """
        Run an evaluation on the benchmark.

        Args:
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens: Maximum number of tokens to generate
            split: Dataset split to evaluate on
            num_workers: Number of parallel workers (defaults to instance value)

        Returns:
            Dictionary with evaluation results
        """
        num_workers = self.num_workers

        # Load dataset
        logger.info(
            f"Loading dataset for {self.benchmark_name} split '{split}'"
        )
        dataset = self.load_dataset(split)

        # Prepare inputs using parallel processing
        system_contents, user_contents, images = self.prepare_inputs(
            dataset, num_workers=num_workers
        )

        # Prepare messages for the API
        messages_list = self.prepare_all_messages(
            system_contents, user_contents, images, num_workers=num_workers
        )

        # Run inference (previously called budget forcing?)
        logger.info(f"Running inference on {len(messages_list)} examples")
        responses = self.inference(
            messages_list=messages_list,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        # 检查所有的repponse不是空字符串
        # if sum(1 for response in responses if response == "") / len(responses) < 0.01:
        #     eval_results = {}
        #     eval_results['detailed_results'] = []
        #     logger.info("More than 1% of responses are empty strings, skipping evaluation")
        # else:
        #     # Evaluate outputs using parallel processing
        #     logger.info(f"Evaluating {len(responses)} outputs")
        #     eval_results = self.evaluate_outputs(
        #         dataset, responses, num_workers=num_workers
        #     )
        logger.info(f"Evaluating {len(responses)} outputs")
        eval_results = self.evaluate_outputs(
            dataset, responses, num_workers=num_workers
        )

        

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_results_dir = os.path.join(
            self.results_dir_base,
            (
                f"{timestamp}_temp_{str(temperature).replace('.', '_')}"
                f"_top_p_{str(top_p).replace('.', '_')}"
                f"_rep_penalty_{str(repetition_penalty).replace('.', '_')}"
            ),
        )
        os.makedirs(current_results_dir, exist_ok=True)

        result_file = os.path.join(current_results_dir, f"result.json")

        # Consolidate results into a single dictionary
        full_results = {
            "hyperparameters": {
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "num_workers": num_workers,
            },
            "timestamp": timestamp,
            "benchmark": self.benchmark_name,
            "model": self.model_name,
            "split": split,
            **eval_results,
        }

        logger.info(f"Saving full results to {result_file}")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Evaluation complete for {self.benchmark_name} on model {self.model_name}"
        )
        logger.info(f"Accuracy: {eval_results.get('accuracy', 'N/A')}")

        return full_results
            