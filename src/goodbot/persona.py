"""GoodBot persona — the main entry point for the Jupyter AI persona."""

import logging
import os
from typing import Any

from jupyter_ai.personas import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

from .agent import build_agent
from .chat_models import ChatLiteLLM
from .config import GoodBotConfig
from .prompt_template import GOODBOT_SYSTEM_PROMPT_TEMPLATE, GoodBotSystemPromptArgs
from .tools import get_all_tools

# File logger for debugging in environments where server logs are inaccessible
_flog = logging.getLogger("goodbot.debug")
_flog.setLevel(logging.DEBUG)
_fh = logging.FileHandler(os.path.expanduser("~/goodbot_debug.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_flog.addHandler(_fh)

GOODBOT_AVATAR_PATH = str(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "static", "goodbot.svg")
    )
)


class GoodBotPersona(BasePersona):
    """AI assistant for clinical data extraction hackathon."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="goodbot",
            avatar_path=GOODBOT_AVATAR_PATH,
            description="Clinical data extraction assistant with document search, web search, and notebook tools.",
            system_prompt="You are GoodBot, an AI assistant for a clinical data extraction hackathon.",
        )

    async def process_message(self, message: Message) -> None:
        _flog.info("process_message: sender=%s body=%r", message.sender, message.body[:100])
        try:
            config = GoodBotConfig()

            # Resolve model ID: use config_manager if available, else config
            if hasattr(self, "config_manager") and self.config_manager and self.config_manager.chat_model:
                model_id = self.config_manager.chat_model
                model_args = self.config_manager.chat_model_args or {}
            else:
                model_id = config.model_id
                model_args = {}

            # gpt-5 family models do not accept the temperature parameter
            supports_temperature = "gpt-5" not in model_id
            model_kwargs = {
                **model_args,
                "model": model_id,
                "streaming": True,
                "max_tokens": config.max_tokens,
            }
            if config.temperature is not None and supports_temperature:
                model_kwargs["temperature"] = config.temperature
            model = ChatLiteLLM(**model_kwargs)

            system_prompt = self._get_system_prompt(
                model_id=model_id, message=message
            )
            tools = get_all_tools()
            agent = await build_agent(
                model, tools, system_prompt, log=self.log
            )

            context = {
                "thread_id": self.ychat.get_id(),
                "username": message.sender,
            }

            async def create_aiter():
                async for token, metadata in agent.astream(
                    {"messages": [{"role": "user", "content": message.body}]},
                    {"configurable": context},
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node", "")
                    if node == "agent" and isinstance(token.content, str) and token.content:
                        yield token.content

            await self.stream_message(create_aiter())

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _flog.error("Error in process_message:\n%s", tb)
            _fh.flush()
            self.send_message(f"Error: {e}\n\n```\n{tb}\n```")

    def _get_system_prompt(
        self, model_id: str, message: Message
    ) -> str:
        context = self.process_attachments(message) or ""
        context = f"User's username is '{message.sender}'\n\n" + context

        args = GoodBotSystemPromptArgs(
            model_id=model_id,
            persona_name=self.name,
            context=context,
        ).model_dump()

        return GOODBOT_SYSTEM_PROMPT_TEMPLATE.render(**args)

    def shutdown(self):
        from .tools.notebook import cleanup_all_kernels
        cleanup_all_kernels()
        super().shutdown()
