import yaml
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    return config

load_config("config/config.yaml")  # Example usage