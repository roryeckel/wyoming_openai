"""Bias speech-to-text toward the names Home Assistant exposes to Assist.

The names of the entities exposed to Assist, with their aliases and the areas and floors they are in, are read over
the Home Assistant websocket API and added to the STT prompt after STT_PROMPT, so a model that has never heard of
"Ecobee" has a reason to write it.
"""

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect

from .const import (
    HASS_CHANGE_DEBOUNCE_SECONDS,
    HASS_NAME_EVENTS,
    HASS_PRIORITY_DOMAINS,
    HASS_REQUEST_TIMEOUT,
    HASS_RETRY_SECONDS,
)
from .utilities import NameGroup, clean_name

_LOGGER = logging.getLogger(__name__)


class HomeAssistantError(Exception):
    """Raised when the names cannot be read from Home Assistant."""


def hass_websocket_url(url: str) -> str:
    """
    The websocket API URL of a Home Assistant instance at a URL such as http://homeassistant.local:8123.

    Raises:
        ValueError: When the URL is not an http or https URL with a host.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(
            f"Home Assistant URL must be an http or https URL such as http://homeassistant.local:8123, got '{url}'"
        )
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, f"{parts.path.rstrip('/')}/api/websocket", "", ""))


def parse_hass_names(
    exposed_ids: Iterable[str],
    states: Iterable[dict[str, Any]],
    entity_entries: dict[str, Any],
    devices: Iterable[dict[str, Any]],
    areas: Iterable[dict[str, Any]],
    floors: Iterable[dict[str, Any]],
) -> tuple[NameGroup, ...]:
    """
    The name groups of the entities exposed to Assist and of the areas and floors, in the order they are added to the
    prompt:

    1. Areas and floors that hold an exposed entity
    2. Entities in HASS_PRIORITY_DOMAINS, which spoken commands usually control
    3. Areas and floors without an exposed entity
    4. Every other exposed entity, mostly sensors

    A reply that does not have this shape raises, and fetch_names reports it as a failure.
    """
    friendly_names = {state["entity_id"]: state["attributes"].get("friendly_name") for state in states}
    device_areas = {device["id"]: device.get("area_id") for device in devices}

    used_area_ids: set[str] = set()
    priority_entities: list[NameGroup] = []
    other_entities: list[NameGroup] = []
    for entity_id in exposed_ids:
        # An entity that is not in the entity registry, such as one set up in YAML without a unique id, has no entry
        entry = entity_entries.get(entity_id) or {}
        if entry.get("disabled_by"):
            continue
        area_id = entry.get("area_id") or device_areas.get(entry.get("device_id"))
        if area_id:
            used_area_ids.add(area_id)

        # The displayed name comes first: the registry name can lack the device's name ("Blinds" for "Bedroom Blinds")
        name = (
            clean_name(friendly_names.get(entity_id))
            or clean_name(entry.get("name"))
            or clean_name(entry.get("original_name"))
        )
        if name is not None:
            domain = entity_id.split(".", 1)[0]
            target = priority_entities if domain in HASS_PRIORITY_DOMAINS else other_entities
            target.append(_name_group(name, entry.get("aliases")))

    used_floor_ids: set[str] = set()
    used_areas: list[NameGroup] = []
    empty_areas: list[NameGroup] = []
    for area in areas:
        if (name := clean_name(area["name"])) is None:
            continue
        used = area["area_id"] in used_area_ids
        if used and area.get("floor_id"):
            used_floor_ids.add(area["floor_id"])
        (used_areas if used else empty_areas).append(_name_group(name, area.get("aliases")))
    for floor in floors:
        if (name := clean_name(floor["name"])) is None:
            continue
        target = used_areas if floor["floor_id"] in used_floor_ids else empty_areas
        target.append(_name_group(name, floor.get("aliases")))

    return (*used_areas, *priority_entities, *empty_areas, *other_entities)


def _name_group(name: str, aliases: Iterable[object] | None) -> NameGroup:
    """A name followed by its aliases, in a stable order since Home Assistant stores aliases as a set."""
    cleaned = {clean_name(alias) for alias in aliases or ()}
    return (name, *sorted((alias for alias in cleaned if alias is not None), key=str.casefold))


class HomeAssistantNames:
    """
    The names Home Assistant exposes to Assist, kept current over one websocket connection shared by every Wyoming
    connection.

    The names are read when the connection opens, and again whenever the entity, device, area or floor registry
    changes, which includes exposing an entity to Assist or hiding it. A connection that fails or closes is opened
    again after a while; meanwhile the names read before are used, or none when none were ever read.
    """

    def __init__(
        self,
        token: str,
        url: str,
        *,
        request_timeout: float = HASS_REQUEST_TIMEOUT,
        retry_seconds: float = HASS_RETRY_SECONDS,
        debounce_seconds: float = HASS_CHANGE_DEBOUNCE_SECONDS,
    ) -> None:
        """
        Raises:
            ValueError: When the URL is not an http or https URL with a host.
        """
        self._token = token
        self.websocket_url = hass_websocket_url(url)
        self._request_timeout = request_timeout
        self._retry_seconds = retry_seconds
        self._debounce_seconds = debounce_seconds
        self._names: tuple[NameGroup, ...] = ()
        self._task: asyncio.Task[None] | None = None

    @property
    def names(self) -> tuple[NameGroup, ...]:
        """The names read last, or none when none were ever read."""
        return self._names

    def start(self) -> None:
        """Connect in the background and keep the names current until aclose()."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="home_assistant_names")

    async def aclose(self) -> None:
        """Close the connection."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        failing = False
        while True:
            try:
                async with connect(self.websocket_url, max_size=None, open_timeout=self._request_timeout) as websocket:
                    async for names in _Session(websocket, self._token, self._request_timeout, self._debounce_seconds):
                        if failing:
                            _LOGGER.info("Reading names from Home Assistant works again")
                            failing = False
                        if names != self._names:
                            # The first names read are worth seeing in the log; later changes only when debugging
                            level = logging.INFO if not self._names else logging.DEBUG
                            _LOGGER.log(level, "Read %d names from Home Assistant", sum(len(group) for group in names))
                        self._names = names
            except Exception as err:
                # Logged as a warning once per outage
                _LOGGER.log(
                    logging.DEBUG if failing else logging.WARNING,
                    "No connection to Home Assistant at %s (%s); using %s, and connecting again in %g seconds",
                    self.websocket_url,
                    _describe(err),
                    "the names read before" if self._names else "no Home Assistant names",
                    self._retry_seconds,
                )
                failing = True
            await asyncio.sleep(self._retry_seconds)


def _describe(err: Exception) -> str:
    """An error as a log reads best: the message of a HomeAssistantError, or the type and message of another."""
    if isinstance(err, HomeAssistantError):
        return str(err)
    return f"{type(err).__name__}: {err}" if str(err) else type(err).__name__


def _changes_names(event: dict[str, Any]) -> bool:
    """
    Whether a registry event can change the names: one that creates or removes an entry, or updates a field the names
    come from, such as a device's firmware version does not. An event of an unexpected shape is taken to.
    """
    fields = HASS_NAME_EVENTS.get(event.get("event_type", ""))
    data = event.get("data") or {}
    changes = data.get("changes")
    if fields is None or data.get("action") != "update" or not isinstance(changes, dict):
        return True
    return not fields.isdisjoint(changes)


class _Session:
    """
    One websocket connection to Home Assistant, iterated for the names: once it opens, then each time the registries
    change. A single reader takes every message in order: command results, and the registry change events it
    subscribes to, which mark the names as changed.
    """

    def __init__(self, websocket: ClientConnection, token: str, request_timeout: float, debounce_seconds: float):
        self._websocket = websocket
        self._token = token
        self._request_timeout = request_timeout
        self._debounce_seconds = debounce_seconds
        self._message_ids = itertools.count(1)
        self._changed = False

    async def __aiter__(self) -> AsyncIterator[tuple[NameGroup, ...]]:
        async with asyncio.timeout(self._request_timeout):
            await self._authenticate()
            for event_type in HASS_NAME_EVENTS:
                await self._command("subscribe_events", event_type=event_type)
        while True:
            self._changed = False
            async with asyncio.timeout(self._request_timeout):
                names = await self._read_names()
            yield names
            if not self._changed:
                await self._receive_event()
            # A change usually comes as several events, such as a new entity and then its exposure; read once they stop
            with contextlib.suppress(TimeoutError):
                while True:
                    async with asyncio.timeout(self._debounce_seconds):
                        await self._receive_event()

    async def _receive(self) -> dict[str, Any]:
        message = json.loads(await self._websocket.recv())
        if message.get("type") == "event" and _changes_names(message.get("event") or {}):
            self._changed = True
        return message

    async def _receive_event(self) -> None:
        """Wait for a registry change that can change the names."""
        self._changed = False
        while not self._changed:
            await self._receive()

    async def _authenticate(self) -> None:
        if (message := await self._receive()).get("type") != "auth_required":
            raise HomeAssistantError(f"Expected auth_required from Home Assistant, got {message.get('type')}")
        await self._websocket.send(json.dumps({"type": "auth", "access_token": self._token}))
        if (message := await self._receive()).get("type") != "auth_ok":
            raise HomeAssistantError(f"Home Assistant rejected the access token: {message.get('message', message)}")

    async def _command(self, command_type: str, **fields: Any) -> Any:
        message_id = next(self._message_ids)
        await self._websocket.send(json.dumps({"id": message_id, "type": command_type, **fields}))
        while (message := await self._receive()).get("id") != message_id or message.get("type") != "result":
            pass  # An event, or the result of another command
        if message.get("success"):
            return message.get("result")
        error = message.get("error") or {}
        if error.get("code") == "unauthorized":
            raise HomeAssistantError(
                f"{command_type} was refused as unauthorized: the access token must belong to a Home Assistant "
                "administrator"
            )
        raise HomeAssistantError(f"{command_type} failed: {error.get('message') or error}")

    async def _read_names(self) -> tuple[NameGroup, ...]:
        exposed = (await self._command("homeassistant/expose_entity/list")).get("exposed_entities") or {}
        exposed_ids = sorted(entity_id for entity_id, assistants in exposed.items() if assistants.get("conversation"))
        states = await self._command("get_states")
        entries = (
            await self._command("config/entity_registry/get_entries", entity_ids=exposed_ids) if exposed_ids else {}
        )
        # Devices are only needed for an entity that takes its area from its device
        needs_devices = any(entry and entry.get("device_id") and not entry.get("area_id") for entry in entries.values())
        devices = await self._command("config/device_registry/list") if needs_devices else []
        areas = await self._command("config/area_registry/list")
        floors = await self._command("config/floor_registry/list")
        return parse_hass_names(exposed_ids, states, entries, devices, areas, floors)
