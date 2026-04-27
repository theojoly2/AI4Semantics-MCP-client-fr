from typing import (
    Any,
)
from os import (
    getenv,
)
from yaml import (
    safe_load,
)
from openai import (
    AsyncOpenAI,
    AsyncAzureOpenAI,
)
from openai.resources.chat.completions import (
    AsyncCompletions,
)


class OpenAIClient():

    def __init__(self) -> None:
        with open('./clients/.openai_client_config.yaml') as file:
            configs: dict[str, Any] = safe_load(file)

        API: str = str(getenv('API'))
        config: dict[str, Any] = configs[API]
        self._config: dict[str, Any] = config

        init = dict(config['init'])
        init['api_key'] = getenv(init['api_key'])

        if config['client'] == 'OpenAI':
            openai: AsyncOpenAI = AsyncOpenAI(**init)

        elif config['client'] == 'AzureOpenAI':
            openai = AsyncAzureOpenAI(**init)

        else:
            raise ValueError('client must be either OpenAI or AzureOpenAI')

        self._openai: AsyncOpenAI = openai
        self.chat_completions: AsyncCompletions = self \
            ._openai \
            .chat \
            .completions
