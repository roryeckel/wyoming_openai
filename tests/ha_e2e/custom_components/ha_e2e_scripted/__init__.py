"""Test-only conversation agent that streams scripted replies through ChatLog."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import ConfigType

from .const import (
    ATTR_DELTA_DELAY_MS,
    ATTR_REPLY,
    DATA_NEXT_REPLY,
    DOMAIN,
    SERVICE_SET_REPLY,
    ScriptedReply,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)


@callback
def _async_register_service(hass: HomeAssistant) -> None:
    """Register the test-only service once."""

    if hass.services.has_service(DOMAIN, SERVICE_SET_REPLY):
        return

    async def async_set_reply(call: ServiceCall) -> None:
        reply = call.data.get(ATTR_REPLY)
        delay = call.data.get(ATTR_DELTA_DELAY_MS, 0)
        if not isinstance(reply, str) or not reply.strip():
            raise HomeAssistantError("set_reply requires a non-empty 'reply' string")
        try:
            delay_ms = int(delay)
        except (TypeError, ValueError) as exc:
            raise HomeAssistantError("delta_delay_ms must be an integer") from exc
        if not 0 <= delay_ms <= 10_000:
            raise HomeAssistantError("delta_delay_ms must be between 0 and 10000")
        hass.data.setdefault(DOMAIN, {})[DATA_NEXT_REPLY] = ScriptedReply(
            text=reply,
            delta_delay_ms=delay_ms,
        )

    hass.services.async_register(DOMAIN, SERVICE_SET_REPLY, async_set_reply)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the service used by pytest."""

    _async_register_service(hass)
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the conversation platform."""

    _async_register_service(hass)
    hass.data.setdefault(DOMAIN, {})
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the conversation platform."""

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
