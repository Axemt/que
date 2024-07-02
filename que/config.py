import toml
import os
from importlib import resources
import que
from typing import Dict, Any, List
from warnings import warn

CONFIG_FOLDER =  os.path.expanduser('~') + '/.config/que'
USR_CONFIG_PATH = f'{CONFIG_FOLDER}/config.toml'
DEFAULT_CONFIG_PATH = 'que/default_config.toml'

def load_default_config(file_needs_creating: bool = False) -> Dict[str, Any]:
    f"""
    Writes a default config file to the user directory {USR_CONFIG_PATH}.
    The default config file is present in the library files in {DEFAULT_CONFIG_PATH}

    Kwargs:
        file_needs_creating: Whether to create the config file if one does not exist in the user directory
    Returns:
        The dict representing the default configuration
    """
    
    with resources.open_text(que, 'default_config.toml') as f:
        default_config = toml.load(f)

    if file_needs_creating:
        with open(USR_CONFIG_PATH, 'w') as f:
            toml.dump(default_config, f)

    return default_config

def __assert_config_format(
        config: Dict[str, Any], 
        expected_fields: List[str], 
        NOT_PRESENT_MSG: str = '{field} was expected in the config file, but was not found'
    ):

    config_keys = config.keys()
    for field in expected_fields:
        assert field in config_keys, NOT_PRESENT_MSG.format(field=field)

def __assert_has_format_fields(
        s: str,
        fields: List[str],
        NOT_PRESENT_MSG: str = '{field} is expected to be a format field of this string, but it was not found:\nstr:{s}'
    ):

    for format_field in fields:
        assert '{' + format_field + '}' in s, NOT_PRESENT_MSG.format(field=format_field, s=s)

def assert_config_format(config: Dict[str, Any]):

    __assert_config_format(config,['verbose', 'documents', 'prompts', 'model'])
    
    __assert_config_format(config['documents'], ['n_documents_per_query'])
    
    __assert_config_format(config['prompts'], ['system_prompt', 'followup_prompt', 'context_template'])
    prompts = config['prompts']
    __assert_has_format_fields(prompts['system_prompt'], ['context'])
    __assert_has_format_fields(prompts['followup_prompt'], ['context'])
    __assert_has_format_fields(prompts['context_template'], ['fname', 'snippet'])
    
    __assert_config_format(config['model'], ['model_id', 'quant'])
    
    

def load_or_create_config() -> Dict[str, Any]:
    f"""
    Loads a config file at {USR_CONFIG_PATH} if it is present and contains expected fields, or loads the default file if not

    Returns
        a valid configuration dict
    """
    is_present = os.path.exists(USR_CONFIG_PATH)
    needs_regen = not is_present

    if is_present:
        try:
            with open(USR_CONFIG_PATH, 'r') as f:
                config = toml.load(f)

            assert_config_format(config)
            needs_regen = False
        except Exception as e:
            # any exception was in the read body or in assert_correct_format.
            # if either fails, reload the default
            warn(
                str(e)
            )
            needs_regen = True


    if needs_regen:
        regen_reason = f'was not found (expected in {USR_CONFIG_PATH})' if not is_present else 'did not contain all required fields'
        warn(
            f'The config file {regen_reason}. Reloading with default file'
        )

        config = load_default_config(file_needs_creating=not is_present)

    return config

QUECONFIG = load_or_create_config()