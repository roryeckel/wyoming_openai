"""Conversation entity that streams scripted replies for the HA E2E harness."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DATA_NEXT_REPLY, DOMAIN, ScriptedReply

PARALLEL_UPDATES = 0
CHUNK_CHARS = 24


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add the single scripted agent entity."""

    async_add_entities([ScriptedConversationEntity(config_entry)])


class ScriptedConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Stream a reply previously supplied through ha_e2e_scripted.set_reply."""

    _attr_name = "HA E2E Scripted"
    _attr_unique_id = "ha_e2e_scripted"
    _attr_supports_streaming = True

    def __init__(self, entry: ConfigEntry) -> None:
        self.entry = entry

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return ["en"]

    async def async_added_to_hass(self) -> None:
        """Register this entity as the conversation agent for its entry."""

        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the agent before the entity goes away."""

        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Stream the next scripted reply through the ChatLog delta API."""

        script: ScriptedReply | None = self.hass.data.get(DOMAIN, {}).pop(DATA_NEXT_REPLY, None)
        if script is None:
            raise HomeAssistantError(
                "No scripted reply is configured; call ha_e2e_scripted.set_reply before the pipeline run"
            )

        async def delta_stream() -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
            yield {"role": "assistant"}
            chunks = [script.text[index : index + CHUNK_CHARS] for index in range(0, len(script.text), CHUNK_CHARS)]
            for index, chunk in enumerate(chunks):
                yield {"content": chunk}
                if script.delta_delay_ms and index < len(chunks) - 1:
                    await asyncio.sleep(script.delta_delay_ms / 1000)

        async for _content in chat_log.async_add_delta_content_stream(self.entity_id, delta_stream()):
            pass

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
