import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import pytest_asyncio
from websockets.asyncio.server import serve

import wyoming_openai.home_assistant as home_assistant_module
from wyoming_openai.const import HASS_CLOSE_TIMEOUT
from wyoming_openai.home_assistant import HomeAssistantNames, _changes_names, hass_websocket_url, parse_hass_names

TOKEN = "secret-token"

EXPOSED = {
    "exposed_entities": {
        "climate.ecobee": {"conversation": True},
        "light.blinds_lamp": {"conversation": True},
        "sensor.outdoor_temperature": {"conversation": True},
        "switch.hidden": {"conversation": False},
        "light.disabled": {"conversation": True},
        "light.yaml_lamp": {"conversation": True},
    }
}
EXPOSED_IDS = ["climate.ecobee", "light.blinds_lamp", "light.disabled", "light.yaml_lamp", "sensor.outdoor_temperature"]
STATES = [
    {"entity_id": "climate.ecobee", "attributes": {"friendly_name": "Ecobee"}},
    {"entity_id": "light.blinds_lamp", "attributes": {"friendly_name": "Bedroom  Lamp"}},
    {"entity_id": "sensor.outdoor_temperature", "attributes": {}},
    {"entity_id": "switch.hidden", "attributes": {"friendly_name": "Hidden"}},
    {"entity_id": "light.yaml_lamp", "attributes": {"friendly_name": "Porch Lamp"}},
]
ENTRIES = {
    "climate.ecobee": {"area_id": "living_room", "aliases": ["Thermostat"]},
    # The device gives this entity its area
    "light.blinds_lamp": {"device_id": "dev1", "area_id": None, "aliases": []},
    "sensor.outdoor_temperature": {"original_name": "Outdoor Temperature", "aliases": []},
    "light.disabled": {"name": "Disabled", "disabled_by": "user", "area_id": "attic"},
    # Not in the entity registry, so Home Assistant answers with null
    "light.yaml_lamp": None,
}
DEVICES = [{"id": "dev1", "area_id": "bedroom"}]
AREAS = [
    {"area_id": "living_room", "name": "Living Room", "aliases": ["Lounge"], "floor_id": "ground"},
    {"area_id": "bedroom", "name": "Bedroom", "aliases": [], "floor_id": "upstairs"},
    {"area_id": "attic", "name": "Attic", "aliases": [], "floor_id": None},
]
FLOORS = [
    {"floor_id": "ground", "name": "Ground Floor", "aliases": []},
    {"floor_id": "upstairs", "name": "Upstairs", "aliases": []},
    {"floor_id": "basement", "name": "Basement", "aliases": []},
]
RESULTS = {
    "homeassistant/expose_entity/list": EXPOSED,
    "get_states": STATES,
    "config/entity_registry/get_entries": ENTRIES,
    "config/device_registry/list": DEVICES,
    "config/area_registry/list": AREAS,
    "config/floor_registry/list": FLOORS,
}
EXPECTED_NAMES = (
    # Areas and floors in use
    ("Living Room", "Lounge"),
    ("Bedroom",),
    ("Ground Floor",),
    ("Upstairs",),
    # Entities in the priority domains
    ("Ecobee", "Thermostat"),
    ("Bedroom Lamp",),
    ("Porch Lamp",),
    # Areas and floors not in use
    ("Attic",),
    ("Basement",),
    # Other entities
    ("Outdoor Temperature",),
)


class FakeHomeAssistant:
    """
    A websocket server that answers like Home Assistant, with a result or an error per command type, and sends a
    registry change event to every connection on change(). after_reply, when set, is awaited with each command after
    its reply is sent.
    """

    def __init__(self):
        self.results = dict(RESULTS)
        self.errors = {}
        self.token = TOKEN
        self.stall = False
        self.commands = []
        self.connections = []
        self.logins = 0
        self.after_reply: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self.url = ""

    def reads(self):
        """How many times the names were read: the first command of each read."""
        return sum(command["type"] == "homeassistant/expose_entity/list" for command in self.commands)

    async def change(self, event_type="area_registry_updated", data=None):
        for websocket, subscriptions in self.connections:
            for subscription_id, subscribed_type in subscriptions.items():
                if subscribed_type == event_type:
                    event = {"event_type": event_type, "data": data or {"action": "create"}}
                    event = {"id": subscription_id, "type": "event", "event": event}
                    await websocket.send(json.dumps(event))

    async def disconnect(self):
        for websocket, _ in self.connections:
            await websocket.close()

    async def handler(self, websocket):
        await websocket.send(json.dumps({"type": "auth_required", "ha_version": "2026.11.0"}))
        auth = json.loads(await websocket.recv())
        self.logins += 1
        if auth.get("access_token") != self.token:
            await websocket.send(json.dumps({"type": "auth_invalid", "message": "Invalid access token or password"}))
            return
        await websocket.send(json.dumps({"type": "auth_ok"}))
        subscriptions = {}
        self.connections.append((websocket, subscriptions))
        try:
            async for raw in websocket:
                message = json.loads(raw)
                self.commands.append(message)
                if self.stall:
                    # Never answers, until the client gives up and closes the connection
                    await websocket.wait_closed()
                    return
                reply = {"id": message["id"], "type": "result"}
                if message["type"] == "subscribe_events":
                    subscriptions[message["id"]] = message["event_type"]
                    reply.update(success=True, result=None)
                elif message["type"] in self.errors:
                    reply.update(success=False, error=self.errors[message["type"]])
                else:
                    reply.update(success=True, result=self.results[message["type"]])
                await websocket.send(json.dumps(reply))
                if self.after_reply is not None:
                    await self.after_reply(message)
        finally:
            self.connections.remove((websocket, subscriptions))


def _port(server):
    return next(iter(server.sockets)).getsockname()[1]


@pytest_asyncio.fixture
async def fake_hass():
    fake = FakeHomeAssistant()
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        fake.url = f"http://127.0.0.1:{_port(server)}"
        yield fake


@pytest_asyncio.fixture
async def watch(fake_hass):
    """Start watching the fake Home Assistant, and stop when the test ends."""
    started = []

    def start(token=TOKEN, **kwargs):
        # Only the tests about them read again without a change, or cut a burst of changes short
        kwargs = {
            "retry_seconds": 0.01,
            "debounce_seconds": 0.01,
            "max_wait_seconds": 60,
            "refresh_seconds": 60,
            **kwargs,
        }
        names = HomeAssistantNames(token, fake_hass.url, **kwargs)
        names.start()
        started.append(names)
        return names

    yield start
    for names in started:
        await names.aclose()


async def until(condition, timeout=5.0):
    """Wait for a condition the background connection makes true; the timeout only bounds a failing test."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://homeassistant.local:8123", "ws://homeassistant.local:8123/api/websocket"),
        ("https://ha.example.com/", "wss://ha.example.com/api/websocket"),
        # A reverse proxy may serve Home Assistant under a path
        ("https://example.com/ha", "wss://example.com/ha/api/websocket"),
        # The REST API URL, as wyoming-faster-whisper's --hass-api takes, or the websocket API URL itself
        ("http://homeassistant.local:8123/api", "ws://homeassistant.local:8123/api/websocket"),
        ("https://example.com/ha/api/", "wss://example.com/ha/api/websocket"),
        ("http://ha:8123/api/websocket", "ws://ha:8123/api/websocket"),
    ],
)
def test_hass_websocket_url(url, expected):
    assert hass_websocket_url(url) == expected


@pytest.mark.parametrize("url", ["ftp://ha", "ws://ha:8123", "homeassistant.local:8123", "http:///"])
def test_hass_websocket_url_rejects_other_urls(url):
    with pytest.raises(ValueError, match="Home Assistant URL"):
        hass_websocket_url(url)
    with pytest.raises(ValueError, match="Home Assistant URL"):
        HomeAssistantNames(TOKEN, url)


def test_parse_hass_names_ranks_exposed_names():
    assert parse_hass_names(EXPOSED_IDS, STATES, ENTRIES, DEVICES, AREAS, FLOORS) == EXPECTED_NAMES


def test_parse_hass_names_takes_turns_by_domain_and_puts_unavailable_entities_last():
    exposed_ids = sorted(
        ["light.a", "light.b", "light.c", "lock.front", "scene.movie", "sensor.power", "switch.kettle", "switch.gone"]
    )
    states = [
        {"entity_id": entity_id, "state": "on", "attributes": {"friendly_name": name}}
        for entity_id, name in [
            ("light.a", "Lamp A"),
            ("light.b", "Lamp B"),
            ("light.c", "Lamp C"),
            ("lock.front", "Front Door"),
            ("scene.movie", "Movie Time"),
            ("sensor.power", "Power"),
            ("switch.kettle", "Kettle"),
        ]
    ]
    states.append({"entity_id": "switch.gone", "state": "unavailable", "attributes": {"friendly_name": "Old Plug"}})

    names = parse_hass_names(exposed_ids, states, {}, [], [], [])

    assert names == (
        # One of each domain, then the second of each, and so on
        ("Lamp A",),
        ("Front Door",),
        ("Movie Time",),
        ("Kettle",),
        ("Lamp B",),
        ("Lamp C",),
        # Other entities
        ("Power",),
        # Unavailable entities last, whatever their domain, so a prompt that runs out of room drops them first
        ("Old Plug",),
    )


def test_parse_hass_names_rejects_an_unexpected_reply():
    with pytest.raises(KeyError):
        parse_hass_names(EXPOSED_IDS, [{"attributes": {}}], ENTRIES, DEVICES, AREAS, FLOORS)


@pytest.mark.asyncio
async def test_reads_names_when_connected(fake_hass, watch, caplog):
    with caplog.at_level(logging.INFO, logger="wyoming_openai.home_assistant"):
        names = watch()
        assert names.names == ()
        await until(lambda: names.names == EXPECTED_NAMES)

    assert "Read 12 names from Home Assistant" in caplog.text
    subscribed = [command["event_type"] for command in fake_hass.commands if command["type"] == "subscribe_events"]
    assert subscribed == [
        "entity_registry_updated",
        "device_registry_updated",
        "area_registry_updated",
        "floor_registry_updated",
        "homeassistant_started",
    ]
    read = [command for command in fake_hass.commands if command["type"] != "subscribe_events"]
    assert [command["type"] for command in read] == list(RESULTS)
    assert read[2]["entity_ids"] == EXPOSED_IDS
    ids = [command["id"] for command in fake_hass.commands]
    assert ids == list(range(1, len(ids) + 1))


@pytest.mark.asyncio
async def test_skips_devices_when_every_entity_has_an_area(fake_hass, watch):
    fake_hass.results["config/entity_registry/get_entries"] = {"climate.ecobee": {"area_id": "living_room"}}

    names = watch()
    await until(lambda: names.names != ())

    assert "config/device_registry/list" not in [command["type"] for command in fake_hass.commands]


@pytest.mark.asyncio
async def test_reads_names_again_when_a_registry_changes(fake_hass, watch):
    names = watch()
    await until(lambda: names.names == EXPECTED_NAMES)

    fake_hass.results["config/area_registry/list"] = [{"area_id": "garage", "name": "Garage", "aliases": []}]
    fake_hass.results["config/floor_registry/list"] = []
    await fake_hass.change("area_registry_updated")

    await until(lambda: ("Garage",) in names.names)
    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_reads_names_again_when_home_assistant_finishes_starting(fake_hass, watch):
    """Test a read made while Home Assistant was starting is made again once it has, as no registry event reports
    the entities that come up meanwhile."""
    names = watch()
    await until(lambda: names.names == EXPECTED_NAMES)

    fake_hass.results["config/area_registry/list"] = [{"area_id": "garage", "name": "Garage", "aliases": []}]
    await fake_hass.change("homeassistant_started")

    await until(lambda: ("Garage",) in names.names)
    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_reads_names_from_a_home_assistant_without_floors(fake_hass, watch):
    """Test Home Assistant before 2024.4, which has no floor registry, gets every name but the floors'."""
    fake_hass.errors["config/floor_registry/list"] = {"code": "unknown_command", "message": "Unknown command."}

    names = watch()
    await until(lambda: names.names != ())

    floors = {("Ground Floor",), ("Upstairs",), ("Basement",)}
    assert names.names == tuple(group for group in EXPECTED_NAMES if group not in floors)
    assert fake_hass.logins == 1


def _update(event_type, *fields):
    return {"event_type": event_type, "data": {"action": "update", "changes": dict.fromkeys(fields)}}


@pytest.mark.parametrize(
    ("event", "changes_names"),
    [
        # An exposure to Assist, a rename, a move to another area
        (_update("entity_registry_updated", "options"), True),
        (_update("entity_registry_updated", "icon", "name"), True),
        (_update("device_registry_updated", "area_id"), True),
        (_update("device_registry_updated", "name_by_user"), True),
        # Fields the names do not come from
        (_update("entity_registry_updated", "icon", "unit_of_measurement"), False),
        (_update("device_registry_updated", "sw_version"), False),
        # Any creation or removal, and any area or floor change
        ({"event_type": "entity_registry_updated", "data": {"action": "create", "entity_id": "light.new"}}, True),
        ({"event_type": "device_registry_updated", "data": {"action": "remove", "device_id": "dev1"}}, True),
        (_update("area_registry_updated", "icon"), True),
        (_update("floor_registry_updated", "level"), True),
        # The end of Home Assistant's startup, when the entities still loading at the read have come up
        ({"event_type": "homeassistant_started", "data": {}}, True),
        # An event of an unexpected shape is taken to change them
        ({"event_type": "entity_registry_updated", "data": {"action": "update"}}, True),
        ({"event_type": "entity_registry_updated"}, True),
    ],
)
def test_changes_names(event, changes_names):
    assert _changes_names(event) is changes_names


@pytest.mark.asyncio
async def test_reads_again_only_for_a_change_to_the_names(fake_hass, watch):
    names = watch()
    await until(lambda: names.names == EXPECTED_NAMES)

    await fake_hass.change("device_registry_updated", _update("", "sw_version")["data"])
    await asyncio.sleep(0.1)
    assert fake_hass.reads() == 1

    await fake_hass.change("entity_registry_updated", _update("", "options")["data"])
    await until(lambda: fake_hass.reads() == 2)


@pytest.mark.asyncio
async def test_reads_once_for_a_burst_of_changes(fake_hass, watch):
    names = watch(debounce_seconds=0.2)
    await until(lambda: names.names == EXPECTED_NAMES)

    for event_type in ("entity_registry_updated", "entity_registry_updated", "device_registry_updated"):
        await fake_hass.change(event_type)
    await until(lambda: fake_hass.reads() == 2)
    await asyncio.sleep(0.3)

    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_a_steady_stream_of_changes_cannot_put_off_the_read(fake_hass, watch):
    names = watch(debounce_seconds=0.2, max_wait_seconds=0.3)
    await until(lambda: names.names == EXPECTED_NAMES)

    # A change every 0.05 seconds never leaves the 0.2 seconds of quiet the debounce waits for
    for _ in range(30):
        await fake_hass.change("entity_registry_updated")
        await asyncio.sleep(0.05)

    assert fake_hass.reads() >= 3


@pytest.mark.asyncio
async def test_reads_again_without_a_change_every_refresh_seconds(fake_hass, watch):
    names = watch(refresh_seconds=0.05)
    await until(lambda: names.names == EXPECTED_NAMES)

    # Such as an entity that became unavailable, which no registry event reports
    fake_hass.results["get_states"] = [
        {**state, "state": "unavailable"} if state["entity_id"] == "climate.ecobee" else state for state in STATES
    ]

    await until(lambda: names.names[-1] == ("Ecobee", "Thermostat"))
    assert fake_hass.reads() >= 2


@pytest.mark.asyncio
async def test_reads_again_after_a_change_during_a_read(fake_hass, watch):
    hallway = {"friendly_name": "Hallway Thermostat"}
    renamed = [{**s, "attributes": hallway} if s["entity_id"] == "climate.ecobee" else s for s in STATES]

    async def change_during_the_first_read(message):
        # The states were already sent with the old name, so only another read can pick up the rename
        if message["type"] == "get_states" and fake_hass.reads() == 1:
            fake_hass.results["get_states"] = renamed
            await fake_hass.change("entity_registry_updated", _update("", "name")["data"])

    fake_hass.after_reply = change_during_the_first_read
    names = watch()

    await until(lambda: ("Hallway Thermostat", "Thermostat") in names.names)
    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_ignores_a_proxy_set_in_the_environment(fake_hass, watch, monkeypatch):
    # A proxy that is not there: reaching Home Assistant through it would fail
    for variable in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9")
    for variable in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(variable, raising=False)

    names = watch()

    await until(lambda: names.names == EXPECTED_NAMES)


@pytest.mark.asyncio
async def test_bounds_the_wait_for_home_assistant_to_close(fake_hass, watch, monkeypatch):
    calls = []
    real_connect = home_assistant_module.connect

    def recording_connect(*args, **kwargs):
        calls.append(kwargs)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(home_assistant_module, "connect", recording_connect)

    names = watch()
    await until(lambda: names.names == EXPECTED_NAMES)

    assert calls[0]["close_timeout"] == HASS_CLOSE_TIMEOUT
    assert calls[0]["proxy"] is None


@pytest.mark.asyncio
async def test_logs_the_first_read_of_every_connection(fake_hass, watch, caplog):
    for command_type in ("homeassistant/expose_entity/list", "config/area_registry/list", "config/floor_registry/list"):
        fake_hass.results[command_type] = {"exposed_entities": {}} if "expose" in command_type else []

    with caplog.at_level(logging.INFO, logger="wyoming_openai.home_assistant"):
        watch()
        # Nothing exposed is still a working connection, and worth seeing
        await until(lambda: "Read 0 names from Home Assistant" in caplog.text)
        await fake_hass.disconnect()
        await until(lambda: caplog.text.count("Read 0 names from Home Assistant") == 2)


@pytest.mark.asyncio
async def test_reconnects_and_keeps_the_names_while_disconnected(fake_hass, watch, caplog):
    names = watch(retry_seconds=0.2)
    await until(lambda: names.names == EXPECTED_NAMES)

    with caplog.at_level(logging.INFO, logger="wyoming_openai.home_assistant"):
        await fake_hass.disconnect()
        await until(lambda: "No connection to Home Assistant" in caplog.text)
        assert names.names == EXPECTED_NAMES
        await until(lambda: caplog.text.count("Read 12 names from Home Assistant") == 1 and fake_hass.reads() == 2)

    assert "using the names read before, and connecting again in 0.2 seconds" in caplog.text
    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_stops_after_a_rejected_token(fake_hass, watch, caplog):
    with caplog.at_level(logging.DEBUG, logger="wyoming_openai.home_assistant"):
        names = watch(token="wrong")
        await until(lambda: "rejected the access token" in caplog.text)
        # Each attempt counts as a failed login, which Home Assistant can answer by banning the address
        await asyncio.sleep(0.1)

    assert fake_hass.logins == 1
    assert names.names == ()
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(errors) == 1
    assert "Home Assistant rejected the access token: Invalid access token or password" in errors[0]
    assert "Not connecting to Home Assistant at" in errors[0]
    assert "using no Home Assistant names" in errors[0]


@pytest.mark.asyncio
async def test_stops_after_a_token_that_is_not_an_administrators(fake_hass, watch, caplog):
    fake_hass.errors["homeassistant/expose_entity/list"] = {"code": "unauthorized", "message": "Unauthorized"}

    with caplog.at_level(logging.DEBUG, logger="wyoming_openai.home_assistant"):
        names = watch()
        await until(lambda: "must belong to a Home Assistant administrator" in caplog.text)
        # Home Assistant logs an error for each command it refuses
        await asyncio.sleep(0.1)

    assert fake_hass.logins == 1
    assert names.names == ()
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(errors) == 1
    assert "Not connecting to Home Assistant at" in errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_type", "error", "message"),
    [
        ("get_states", {"code": "unknown_error", "message": "Boom"}, "get_states failed: Boom"),
        # A command only an optional read may be missing
        ("get_states", {"code": "unknown_command", "message": "Unknown command."}, "Unknown command."),
    ],
)
async def test_reports_a_failed_command(fake_hass, watch, caplog, command_type, error, message):
    fake_hass.errors[command_type] = error

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.home_assistant"):
        names = watch()
        await until(lambda: message in caplog.text)

    assert names.names == ()


@pytest.mark.asyncio
async def test_reports_an_unexpected_reply(fake_hass, watch, caplog):
    fake_hass.results["config/area_registry/list"] = [{"name": "No id"}]

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.home_assistant"):
        watch()
        await until(lambda: "KeyError: 'area_id'" in caplog.text)


@pytest.mark.asyncio
async def test_reports_a_server_that_does_not_answer(fake_hass, watch, caplog):
    fake_hass.stall = True

    with caplog.at_level(logging.WARNING, logger="wyoming_openai.home_assistant"):
        watch(request_timeout=0.05)
        await until(lambda: "(TimeoutError)" in caplog.text)


@pytest.mark.asyncio
async def test_reports_an_unreachable_server(caplog):
    async def close(websocket):
        await websocket.close()

    async with serve(close, "127.0.0.1", 0) as server:
        port = _port(server)
    names = HomeAssistantNames(TOKEN, f"http://127.0.0.1:{port}", retry_seconds=10)
    with caplog.at_level(logging.WARNING, logger="wyoming_openai.home_assistant"):
        names.start()
        try:
            await until(lambda: "No connection to Home Assistant" in caplog.text)
        finally:
            await names.aclose()


@pytest.mark.asyncio
async def test_close_ends_the_connection(fake_hass, watch):
    names = watch()
    await until(lambda: names.names == EXPECTED_NAMES)

    await names.aclose()

    await until(lambda: not fake_hass.connections)
