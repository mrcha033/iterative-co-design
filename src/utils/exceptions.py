"""
Custom exceptions for the iterative co-design framework.
"""


class IterativeCoDesignError(Exception):
    """Base exception for the iterative co-design framework."""
    pass


class ModelNotSupportedError(IterativeCoDesignError):
    """Raised when a model is not supported."""
    
    def __init__(self, model_name: str, supported_models: list):
        self.model_name = model_name
        self.supported_models = supported_models
        super().__init__(
            f"Model '{model_name}' not supported. "
            f"Supported models: {', '.join(supported_models)}"
        )


class LayerNotFoundError(IterativeCoDesignError):
    """Raised when a layer is not found in a model."""
    
    def __init__(self, layer_name: str, model_name: str, available_layers: list):
        self.layer_name = layer_name
        self.model_name = model_name
        self.available_layers = available_layers
        
        # Show first 10 layers to avoid overwhelming output
        displayed_layers = available_layers[:10]
        layer_list = ', '.join(displayed_layers)
        if len(available_layers) > 10:
            layer_list += f" (showing first 10 of {len(available_layers)} layers)"
        
        super().__init__(
            f"Layer '{layer_name}' not found in model '{model_name}'. "
            f"Available layers: {layer_list}"
        )


class DatasetNotSupportedError(IterativeCoDesignError):
    """Raised when a dataset is not supported."""
    
    def __init__(self, dataset_name: str, supported_datasets: list):
        self.dataset_name = dataset_name
        self.supported_datasets = supported_datasets
        super().__init__(
            f"Dataset '{dataset_name}' not supported. "
            f"Supported datasets: {', '.join(supported_datasets)}"
        )


class InvalidPermutationError(IterativeCoDesignError):
    """Raised when a permutation is invalid."""
    
    def __init__(self, message: str, permutation_size: int = None, expected_size: int = None):
        self.permutation_size = permutation_size
        self.expected_size = expected_size
        
        if permutation_size is not None and expected_size is not None:
            message += f" (got {permutation_size}, expected {expected_size})"
        
        super().__init__(message)


class ConfigurationError(IterativeCoDesignError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_field: str = None):
        self.config_field = config_field
        if config_field:
            message = f"Configuration error in '{config_field}': {message}"
        super().__init__(message)


class ModelLoadError(IterativeCoDesignError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, error_message: str):
        self.model_name = model_name
        self.error_message = error_message
        super().__init__(f"Failed to load model '{model_name}': {error_message}")


class DatasetLoadError(IterativeCoDesignError):
    """Raised when dataset loading fails."""
    
    def __init__(self, dataset_name: str, error_message: str):
        self.dataset_name = dataset_name
        self.error_message = error_message
        super().__init__(f"Failed to load dataset '{dataset_name}': {error_message}")


class PermutationApplicationError(IterativeCoDesignError):
    """Raised when permutation application fails."""
    
    def __init__(self, layer_name: str, error_message: str):
        self.layer_name = layer_name
        self.error_message = error_message
        super().__init__(f"Failed to apply permutation to layer '{layer_name}': {error_message}")


class ProfilingError(IterativeCoDesignError):
    """Raised when profiling fails."""
    
    def __init__(self, message: str, tool: str = None):
        self.tool = tool
        if tool:
            message = f"Profiling error with {tool}: {message}"
        super().__init__(message)


def create_helpful_error_message(error: Exception, context: str = None) -> str:
    """
    Create a helpful error message with context and suggestions.
    
    Args:
        error: The original exception
        context: Additional context about what was being done
        
    Returns:
        Formatted error message with suggestions
    """
    message = f"ERROR: {str(error)}"
    
    if context:
        message = f"{context}\n{message}"
    
    # Add suggestions based on error type
    if isinstance(error, ModelNotSupportedError):
        message += "\n\nSuggestions:"
        message += "\n- Check the model name spelling"
        message += "\n- Use one of the supported models listed above"
        message += "\n- If you need a custom model, implement it in src/models/"
    
    elif isinstance(error, LayerNotFoundError):
        message += "\n\nSuggestions:"
        message += "\n- Check the layer name spelling"
        message += "\n- Use model.get_layer_names() to see all available layers"
        message += "\n- Layer names are case-sensitive"
        message += "\n- For nested layers, use dot notation (e.g., 'layers.0.mixer')"
    
    elif isinstance(error, DatasetNotSupportedError):
        message += "\n\nSuggestions:"
        message += "\n- Check the dataset name spelling"
        message += "\n- Use one of the supported datasets listed above"
        message += "\n- If you need a custom dataset, implement it in src/utils/"
    
    elif isinstance(error, InvalidPermutationError):
        message += "\n\nSuggestions:"
        message += "\n- Ensure permutation is a valid rearrangement of indices"
        message += "\n- Check that permutation size matches layer dimension"
        message += "\n- Permutation should contain each index exactly once"
    
    elif isinstance(error, ConfigurationError):
        message += "\n\nSuggestions:"
        message += "\n- Check your config.yaml file for syntax errors"
        message += "\n- Validate configuration values against allowed options"
        message += "\n- Use the default config as a template"
    
    elif isinstance(error, (ModelLoadError, DatasetLoadError)):
        message += "\n\nSuggestions:"
        message += "\n- Check internet connection for downloading"
        message += "\n- Verify available disk space"
        message += "\n- Try clearing cache and downloading again"
        message += "\n- Check that required dependencies are installed"
    
    return message