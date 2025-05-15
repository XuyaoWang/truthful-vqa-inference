import os
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from truthful_vqa_inference.utils.uuid import generate_hash_uid

logger = logging.getLogger(__name__)

def cached_requests(
    messages: List[Dict[str, Any]],
    model: str,
    max_completion_tokens: int = 4096,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    top_p: float = 0.9,
    api_key: Optional[str] = None,
    api_base: str = None,
    cache_dir: str = './cache',
    max_try: int = 3,
    timeout: int = 1200,
) -> str:
    """
    Make API requests with caching to avoid duplicate calls.
    
    Args:
        messages: List of message dictionaries to send to the API
        model: Model name to use for completion
        max_completion_tokens: Maximum number of tokens in the completion
        temperature: Sampling temperature (higher = more random)
        repetition_penalty: Penalty for token repetition
        top_p: Nucleus sampling parameter
        api_key: API key for authentication (if empty, will check environment variables)
        api_base: Base URL for API endpoint (if empty, will check environment variables)
        cache_dir: Directory to store cache files
        max_try: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        The completed text from the API response
    """
        
    if not api_key:
        api_key = os.environ.get("API_KEY", "")
    
    if not api_base:
        api_base = os.environ.get("API_BASE", "https://api.openai.com/v1")

    if not api_key:
        raise ValueError("API key is not provided")
        
    uuid = generate_hash_uid({
        'messages': messages,
        'max_completion_tokens': max_completion_tokens,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_p': top_p,
        'model': model,
    })

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{uuid}.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            try:
                result = json.load(f)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Invalid cache file {cache_path}, removing it")
                os.remove(cache_path)
    # else:
    #     return ''

    while max_try > 0:
        try:
            headers = {"Content-Type": "application/json", "Connection": "close"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            response = requests.post(
                api_base,
                headers=headers,
                json={
                    "model": model,
                    "max_completion_tokens": max_completion_tokens,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                },
                timeout=timeout,
            )
            
            if response.status_code == 200:
                response_text = response.json()['choices'][0]['message']['content']
                logger.info(f"API request succeeded, received {len(response_text)} chars")
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(response_text, f, ensure_ascii=False)
                    
                return response_text
            elif response.status_code == 400:
                err_msg = f"API error, status code: {response.status_code}\nresponse: {response.json()['error']['message']}"
                logger.error(err_msg)
                return response.json()['error']['message']
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                time.sleep(3)
                max_try -= 1
                continue
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            time.sleep(3)
            max_try -= 1
            continue
    
    logger.error('API failed after all retries')
    return ''
