"""
Open Source Model Inference Engine.
"""

from typing import List, Dict, Any, Optional
from .base import BaseInferenceEngine
from .engines.gemma_engine import GemmaInferenceEngine
from .engines.qwen_engine import QwenInferenceEngine


class OpenSourceInferenceEngine:
    """Factory class for open source inference engines."""
    
    @staticmethod
    def create_engine(model_type: str, model_path: str, **kwargs) -> BaseInferenceEngine:
        """
        Create an appropriate inference engine based on model type.
        
        Args:
            model_type: Type of model ('qwen2.5vl', etc.)
            model_path: Path to the model
            **kwargs: Additional configuration
            
        Returns:
            Configured inference engine
        """
        if model_type.lower() in ['qwen2.5vl', 'qwen', 'qwen2.5-vl']:
            return QwenInferenceEngine(model_path, **kwargs)
        elif model_type.lower() in ['gemma4', 'gemma-4', 'gemma']:
            return GemmaInferenceEngine(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def list_supported_models() -> List[str]:
        """List supported model types."""
        return ['qwen2.5vl', 'qwen', 'qwen2.5-vl', 'gemma4', 'gemma-4', 'gemma']
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """Get information about a supported model type."""
        model_info = {
            'qwen2.5vl': {
                'name': 'Qwen2.5-VL',
                'description': 'Vision-Language model by Alibaba',
                'supported_backends': ['transformers', 'vllm'],
                'recommended_backend': 'transformers',
                'base_model': 'Qwen/Qwen2.5-VL-3B-Instruct'
            },
            'gemma4': {
                'name': 'Gemma 4',
                'description': 'Multimodal open model family by Google DeepMind',
                'supported_backends': ['transformers'],
                'recommended_backend': 'transformers',
                'base_model': 'google/gemma-4-E2B-it'
            }
        }
        
        normalized = model_type.lower()
        if normalized in ['qwen', 'qwen2.5-vl']:
            normalized = 'qwen2.5vl'
        if normalized in ['gemma-4', 'gemma']:
            normalized = 'gemma4'
        return model_info.get(normalized, {
            'name': 'Unknown',
            'description': 'Model not found',
            'supported_backends': [],
            'recommended_backend': None
        })
