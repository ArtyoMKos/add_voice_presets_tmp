import logging
import os
import configparser


class ConfigWrapper:
    def __init__(
        self,
        config_file: str = "configs/configs.{environment}.ini",
    ) -> None:
        """
        Initialize the ConfigWrapper with a configuration file.

        :param config_file: Path template for the configuration file.
                            The placeholder `{environment}` will be replaced with the value of the
                            `ENVIRONMENT` environment variable or 'local' if not set.
        :raises FileNotFoundError: If the configuration file does not exist.

        .. important::
            Environment variables will replace configuration values in all sections except DEFAULT.
        """
        config_path = config_file.format(environment=os.getenv("ENVIRONMENT", "local"))

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        self.config.read(config_path)
        self._load_env_vars()

    def _load_env_vars(self) -> None:
        """
        Load environment variables into the configuration.

        Environment variables should follow the format `env_{section}_{key}`, where `{section}` is
        the name of the configuration section and `{key}` is the name of the key within that
        section.
        If an environment variable is set, it will override the corresponding configuration value.
        """
        for section in self.config.sections():
            for key in self.config[section]:
                env_value = os.getenv(f"env_{section}_{key}")
                if env_value is not None:
                    logging.debug(f"Replacing value of {section}:{key}.")
                    self.config[section][key] = env_value
