import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

# VLLM specific imports
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

# Base class import
from truthful_vqa_inference.server.base_server import BaseServer
from truthful_vqa_inference.server.registry import ServerRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@ServerRegistry.register(name="vllm")
class VLLMServer(BaseServer):
    """Manages a vLLM server instance using the BaseServer framework."""

    def __init__(self, 
                 # Pass through common args to BaseServer
                 model_path: str, port: int = 8000, host: str = '0.0.0.0',
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_seq_len: int = 32768,
                 dtype: str = 'auto', server_init_timeout: int = 300,
                 model_name: Optional[str] = None,
                 # vLLM Specific Args (add more as needed)
                 limit_mm_per_prompt: Optional[str] = None, 
                 chat_template: Optional[str] = None,
                 enable_prefix_caching: bool = True,
                 disable_log_stats: bool = True,
                 disable_log_requests: bool = True,
                 disable_fastapi_docs: bool = True,
                 uvicorn_log_level: str = "warning",
                 **kwargs): # Catch other potential vLLM args
        """
        Initialize a VLLMServer instance.

        Args:
            model_path, port, host, ... : Common server arguments (see BaseServer).
            limit_mm_per_prompt: Limit on multimedia per prompt (vLLM specific, e.g., "image=10,video=10").
            chat_template: Path to chat template file (vLLM specific).
            enable_prefix_caching, disable_log_stats, ... : Other vLLM specific settings.
            **kwargs: Additional arguments passed directly to vLLM argument parser.
        """
        
        # Prepare framework-specific args for BaseServer
        framework_args = {
            'limit_mm_per_prompt': limit_mm_per_prompt,
            'chat_template': chat_template,
            'enable_prefix_caching': enable_prefix_caching,
            'disable_log_stats': disable_log_stats,
            'disable_log_requests': disable_log_requests,
            'disable_fastapi_docs': disable_fastapi_docs,
            'uvicorn_log_level': uvicorn_log_level,
            'trust_remote_code': True,
            # Add any other vLLM specific args explicitly or pass via kwargs
        }
        framework_args.update(kwargs) # Merge explicit and kwargs

        super().__init__(
            model_path=model_path,
            port=port,
            host=host,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_seq_len=max_seq_len,
            dtype=dtype,
            server_init_timeout=server_init_timeout,
            model_name=model_name,
            # Pass health check path expected by vLLM
            health_check_path="/health", 
            # Pass API endpoint expected by vLLM OpenAI integration
            api_endpoint_path="/v1/chat/completions", 
            framework_specific_args=framework_args
        )
        
        # Store the server task for shutdown
        self._server_task: Optional[asyncio.Task] = None


    def _build_server_args(self) -> Dict[str, Any]:
        """Builds the argument dictionary compatible with vLLM's parser."""
        # Start with common args, already processed by BaseServer
        args = self.common_args.copy()
        # Add framework-specific args stored during __init__
        args.update(self.framework_args)

        # --- Handle vLLM Specific Argument Formatting --- 

        # Format served_model_name (already a list in common_args)
        # args['served_model_name'] = args.get('served_model_name', [args.get('model')])

        # Format limit_mm_per_prompt string into dict
        limit_mm_str = args.pop('limit_mm_per_prompt', None)
        if limit_mm_str:
            limit_dict = {}
            try:
                for item in limit_mm_str.split(','):
                    key, value = item.strip().split('=')
                    limit_dict[key.strip()] = int(value.strip())
                args['limit_mm_per_prompt'] = limit_dict
            except ValueError as e:
                 logger.error(f"Invalid format for limit_mm_per_prompt: '{limit_mm_str}'. Error: {e}. Ignoring.")
                 args['limit_mm_per_prompt'] = None # Or set to default empty dict if needed
        else:
            # Ensure the key exists, potentially with a default if required by vLLM
            args['limit_mm_per_prompt'] = None # or {} if vLLM expects the key

        # Handle chat_template (already correct format if provided)
        if 'chat_template' not in args:
            args['chat_template'] = None # Ensure key exists if needed

        args['max_num_seqs'] = 32
        
        logger.debug(f"Built vLLM server args: {args}")
        return args

    @staticmethod
    async def _start_server_instance(server_args: Dict[str, Any]) -> asyncio.Task:
        """Starts the vLLM OpenAI API server instance."""
        parser = FlexibleArgumentParser(
            description="Run the vLLM OpenAI-compatible API server.")
        parser = make_arg_parser(parser)
        
        parsed_args = parser.parse_args([]) 
        
        unknown_args = []
        for key, value in server_args.items():
             if hasattr(parsed_args, key):
                 setattr(parsed_args, key, value)
             else:
                 if isinstance(value, bool) and value:
                     unknown_args.append(f'--{key.replace("_", "-")}')
                 elif value is not None:
                     unknown_args.append(f'--{key.replace("_", "-")}')
                     unknown_args.append(str(value))
                 logger.warning(f"Argument '{key}' not directly recognized by vLLM parser, passing as unknown arg.")

        # Reparse with unknown args if any
        if unknown_args:
             logger.info(f"Reparsing with unknown args: {unknown_args}")
             parsed_args = parser.parse_args(unknown_args, namespace=parsed_args)

        try:
            validate_parsed_serve_args(parsed_args)
        except Exception as e:
            logger.error(f"vLLM argument validation failed: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"vLLM argument validation failed: {e}") from e

        logger.info("Starting vLLM run_server...")
        server_task = asyncio.create_task(run_server(parsed_args))
        return server_task

    @staticmethod
    async def _shutdown_server_instance(server_instance: asyncio.Task):
        """Shuts down the vLLM server task."""
        if not server_instance or server_instance.done():
            logger.info("vLLM server task already stopped or not found.")
            return

        logger.info("Attempting graceful shutdown of vLLM server task...")
        server_instance.cancel()
        try:
            # Wait for the task to acknowledge cancellation and potentially clean up
            await asyncio.wait_for(server_instance, timeout=10.0) 
        except asyncio.CancelledError:
            logger.info("vLLM server task cancelled successfully.")
        except asyncio.TimeoutError:
             logger.warning("Timeout waiting for vLLM server task to cancel. It might not have shut down cleanly.")
        except Exception as e:
            logger.error(f"Exception during vLLM server task shutdown: {e}")
            logger.error(traceback.format_exc())
        
        logger.info("vLLM server shutdown process completed.")


# --- Example Usage --- 
async def start_server_main():
    """Example usage of the VLLMServer class."""
    # Example configuration
    server = VLLMServer(
        # BaseServer args
        model_path="Qwen/Qwen2.5-VL-3B-Instruct", 
        model_name="qwen-vl-3b", 
        port=8001,
        host="0.0.0.0",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.7, 
        # vLLM specific args (passed via __init__)
        limit_mm_per_prompt="image=10,video=10", 
        chat_template=None, 
        enable_prefix_caching=True, 
        dtype="bfloat16", 
        disable_log_stats=True, 
        disable_log_requests=True, 
        disable_fastapi_docs=True, 
        uvicorn_log_level="warning", 
        server_init_timeout=300
    )
    
    try:
        api_base = await server.start()
        logger.info(f"VLLM Server started via BaseServer at {api_base} (PID: {server.get_pid()})")
        
        # Keep server running - in a real app, this might be run forever
        # or managed by a larger application lifecycle.
        logger.info("Server running. Waiting 60 seconds before stopping...")
        await asyncio.sleep(60)
        logger.info("Finished waiting.")

    except Exception as e:
         logger.error(f"Error during server lifecycle: {e}")
         logger.error(traceback.format_exc())
    finally:
        if server.is_running():
            logger.info("Stopping server...")
            await server.stop()
            logger.info("Server stop initiated.")
        else:
            logger.info("Server not running at the end.")


if __name__ == "__main__":
    try:
        asyncio.run(start_server_main())
    except KeyboardInterrupt:
         logger.info("Main process interrupted by keyboard.")
    finally:
         logger.info("Example script finished.")
