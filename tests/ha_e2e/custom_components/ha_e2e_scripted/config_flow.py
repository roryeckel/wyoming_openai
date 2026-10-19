"""Config flow for the test-only HA E2E scripted conversation agent."""

from __future__ import annotations

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import DOMAIN


class HaE2EScriptedConfigFlow(ConfigFlow, domain=DOMAIN):
    """Create the single test entry."""

    VERSION = 1

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        """Create the entry without asking for any input."""

        if self._async_current_entries():
            return self.async_abort(reason="already_configured")
        return self.async_create_entry(title="HA E2E Scripted Replies", data={})
