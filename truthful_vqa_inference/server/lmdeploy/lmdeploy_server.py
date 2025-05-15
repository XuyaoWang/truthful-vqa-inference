import asyncio
import logging
import traceback
import nest_asyncio
from typing import Any, Dict, List, Optional, Union, Literal

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.openai.api_server import serve

from truthful_vqa_inference.server.base_server import BaseServer
from truthful_vqa_inference.server.registry import ServerRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@ServerRegistry.register(name="lmdeploy")
class LMDeployServer(BaseServer):
    """Manages an LMDeploy server instance using the BaseServer framework."""

    def __init__(self, 
                 model_path: str, port: int = 8000, host: str = '0.0.0.0',
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_seq_len: int = 32768,
                 dtype: str = 'auto', server_init_timeout: int = 300,
                 model_name: Optional[str] = None,
                 # LMDeploy Specific Args
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                #  backend: Literal['turbomind', 'pytorch'] = 'pytorch',
                 backend_config: Optional[Union[PytorchEngineConfig, TurbomindEngineConfig]] = None,
                 chat_template: Optional[ChatTemplateConfig] = None,
                 allow_origins: List[str] = ['*'],
                 allow_credentials: bool = True,
                 allow_methods: List[str] = ['*'],
                 allow_headers: List[str] = ['*'],
                 log_level: str = 'ERROR',
                 api_keys: Optional[Union[List[str], str]] = None,
                 ssl: bool = False,
                 proxy_url: Optional[str] = None,
                 max_log_len: Optional[int] = None,
                 disable_fastapi_docs: bool = True,
                 max_concurrent_requests: Optional[int] = None,
                 reasoning_parser: Optional[str] = None,
                 tool_call_parser: Optional[str] = None,
                 **kwargs):
        """
        Initialize an LMDeployServer instance.

        Args:
            model_path, port, host, ... : Common server arguments (see BaseServer).
            backend: Either 'turbomind' or 'pytorch' backend.
            backend_config: LMDeploy backend configuration instance.
            chat_template: Chat template configuration.
            allow_origins, allow_credentials, allow_methods, allow_headers: CORS settings.
            log_level: Set log level (CRITICAL, ERROR, WARNING, INFO, DEBUG).
            api_keys: Optional list of API keys or single API key string.
            ssl: Enable SSL (requires OS Environment variables SSL_KEYFILE and SSL_CERTFILE).
            proxy_url: The proxy URL to register the api_server.
            max_log_len: Max number of prompt characters/tokens in log.
            disable_fastapi_docs: Whether to disable FastAPI docs endpoints.
            max_concurrent_requests: Maximum number of concurrent requests.
            reasoning_parser: The reasoning parser name.
            tool_call_parser: The tool call parser name.
            **kwargs: Additional arguments passed to LMDeploy serve function.
        """
        
        framework_args = {
            'backend': backend,
            'backend_config': backend_config,
            'chat_template_config': chat_template,
            'allow_origins': allow_origins,
            'allow_credentials': allow_credentials,
            'allow_methods': allow_methods,
            'allow_headers': allow_headers,
            'log_level': log_level,
            'api_keys': api_keys,
            'ssl': ssl,
            'proxy_url': proxy_url,
            'max_log_len': max_log_len,
            'disable_fastapi_docs': disable_fastapi_docs,
            'max_concurrent_requests': max_concurrent_requests,
            'reasoning_parser': reasoning_parser,
            'tool_call_parser': tool_call_parser,
        }
        framework_args.update(kwargs)  # Merge explicit and kwargs

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
            # LMDeploy uses /health for health checks
            health_check_path="/health",
            # LMDeploy's chat completions API path
            api_endpoint_path="/v1/chat/completions",
            framework_specific_args=framework_args
        )
        
        # Store server task for shutdown
        self._serve_task = None

    def _build_server_args(self) -> Dict[str, Any]:
        """Builds the argument dictionary compatible with LMDeploy's serve function."""
        # Start with common args
        args = {
            'model_path': self.common_args['model'],
            'model_name': self.model_name,
            'server_port': self.common_args['port'],
            'server_name': self.common_args['host'],
        }

        if self.framework_args.get('chat_template_config', None) is not None:
            args['chat_template_config'] = ChatTemplateConfig(self.framework_args['chat_template_config'])

        # Get backend type (pytorch or turbomind)
        backend = self.framework_args.get('backend', 'turbomind')
        
        # Handle backend_config
        if 'backend_config' in self.framework_args and self.framework_args['backend_config'] is not None:
            # Use the provided backend_config directly
            args['backend_config'] = self.framework_args['backend_config']
        else:
            # Create new config based on backend type
            if backend == 'pytorch':
                config = PytorchEngineConfig()
            else:  # default to turbomind
                config = TurbomindEngineConfig()
            
            # Apply common configuration parameters
            if self.common_args.get('tensor_parallel_size', 1) > 1:
                config.tp = self.common_args['tensor_parallel_size']
            
            if 'max_seq_len' in self.common_args and self.common_args['max_seq_len'] is not None:
                config.session_len = self.common_args['max_seq_len']
                
            if 'dtype' in self.common_args and self.common_args['dtype'] != 'auto':
                config.dtype = self.common_args['dtype']
                
            if 'gpu_memory_utilization' in self.common_args:
                config.cache_max_entry_count = self.common_args['gpu_memory_utilization']
                
            args['backend_config'] = config
            
        ignored_keys = ['backend_config', 'chat_template_config']

        for key, value in self.framework_args.items():
            if key not in ignored_keys and value is not None:
                args[key] = value
                
        logger.debug(f"Built LMDeploy server args: {args}")
        return args

    @staticmethod
    async def _start_server_instance(server_args: Dict[str, Any]) -> asyncio.Task:
        """Starts the LMDeploy server instance."""
        logger.info("Starting LMDeploy serve task...")

        async def run_lmdeploy_serve():
            
            nest_asyncio.apply() 

            try:
                serve(**server_args)
            except Exception as e:
                logger.error(f"Error in LMDeploy serve: {e}")
                logger.error(traceback.format_exc())
                raise
                
        serve_task = asyncio.create_task(run_lmdeploy_serve())
        return serve_task

    @staticmethod
    async def _shutdown_server_instance(server_instance: asyncio.Task):
        """Shuts down the LMDeploy server task."""
        if not server_instance or server_instance.done():
            logger.info("LMDeploy server task already stopped or not found.")
            return

        logger.info("Attempting graceful shutdown of LMDeploy server task...")
        server_instance.cancel()
        try:
            # Wait for the task to acknowledge cancellation
            await asyncio.wait_for(server_instance, timeout=10.0) 
        except asyncio.CancelledError:
            logger.info("LMDeploy server task cancelled successfully.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for LMDeploy server task to cancel.")
        except Exception as e:
            logger.error(f"Exception during LMDeploy server task shutdown: {e}")
            logger.error(traceback.format_exc())
        
        logger.info("LMDeploy server shutdown process completed.")


async def start_server_main():
    """Example usage of the LMDeployServer class."""
    server = LMDeployServer(
        model_path="OpenGVLab/InternVL3-1B", 
        model_name="internvl3-1b", 
        host="0.0.0.0",
        tensor_parallel_size=4,
        backend="turbomind",
        disable_fastapi_docs=True,
        log_level="ERROR"
    )
    
    try:
        api_base = await server.start()
        logger.info(f"LMDeploy Server started at {api_base} (PID: {server.get_pid()})")
        
        # Keep server running for testing
        logger.info("Server running. Press Ctrl+C to stop...")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping server...")
    except Exception as e:
        logger.error(f"Error during server lifecycle: {e}")
        logger.error(traceback.format_exc())
    finally:
        if server.is_running():
            logger.info("Stopping server...")
            await server.stop()
            logger.info("Server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(start_server_main())
    except KeyboardInterrupt:
        logger.info("Main process interrupted by keyboard.")
    finally:
        logger.info("Script finished.")