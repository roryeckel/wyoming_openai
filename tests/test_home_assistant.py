import asyncio
import json
import logging

import pytest
import pytest_asyncio
from websockets.asyncio.server import serve

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
    registry change event to every connection on change().
    """

    def __init__(self):
        self.results = dict(RESULTS)
        self.errors = {}
        self.token = TOKEN
        self.stall = False
        self.commands = []
        self.connections = []
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
        kwargs = {"retry_seconds": 0.01, "debounce_seconds": 0.01, **kwargs}
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
async def test_reconnects_and_keeps_the_names_while_disconnected(fake_hass, watch, caplog):
    names = watch(retry_seconds=0.2)
    await until(lambda: names.names == EXPECTED_NAMES)

    with caplog.at_level(logging.INFO, logger="wyoming_openai.home_assistant"):
        await fake_hass.disconnect()
        await until(lambda: "No connection to Home Assistant" in caplog.text)
        assert names.names == EXPECTED_NAMES
        await until(lambda: "works again" in caplog.text)

    assert "using the names read before, and connecting again in 0.2 seconds" in caplog.text
    assert fake_hass.reads() == 2


@pytest.mark.asyncio
async def test_reports_a_rejected_token_once_and_keeps_trying(fake_hass, watch, caplog):
    with caplog.at_level(logging.DEBUG, logger="wyoming_openai.home_assistant"):
        names = watch(token="wrong")
        await until(lambda: caplog.text.count("No connection to Home Assistant") >= 3)
        fake_hass.token = "wrong"
        await until(lambda: names.names == EXPECTED_NAMES)

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Home Assistant rejected the access token: Invalid access token or password" in warnings[0]
    assert "using no Home Assistant names" in warnings[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_type", "error", "message"),
    [
        (
            "homeassistant/expose_entity/list",
            {"code": "unauthorized", "message": "Unauthorized"},
            "must belong to a Home Assistant administrator",
        ),
        ("get_states", {"code": "unknown_error", "message": "Boom"}, "get_states failed: Boom"),
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
