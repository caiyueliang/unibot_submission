"""Websocket transport for driving a policy across a network.

A *policy* here is any object exposing three members:

    policy.metadata           -> dict describing the obs / action contract
    policy.get_action(obs)    -> dict of action chunks for one observation
    policy.reset()            -> drop whatever per-episode state it holds

``PolicyService`` wraps such an object and publishes it over websocket; the
compute-heavy side can then run on one machine while ``RemotePolicy`` connects
from another and is used exactly like a local policy. Numpy arrays and scalars
cross the wire as msgpack via a small custom hook; camera images are additionally
JPEG-compressed so a stack of frames does not have to travel as raw pixels.
"""

from __future__ import annotations

import asyncio
import http
import io
import logging
import time
import traceback

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.server import serve as _open_server
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from websockets.sync.client import connect as _open_client

__all__ = ["PolicyService", "RemotePolicy"]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Numpy <-> msgpack                                                           #
# --------------------------------------------------------------------------- #
#
# msgpack has no built-in representation for numpy values, so a set of hooks
# bridges the gap:
#
#   * Camera images -- uint8 arrays laid out channels-last, i.e. any
#     ``(..., H, W, C)`` stack with ``C in (1, 3)`` -- are JPEG-encoded. Each
#     frame of the (possibly time-stacked) batch becomes one JPEG blob, and the
#     original shape rides alongside so the exact stack is rebuilt on the far
#     side. JPEG is *lossy*: this trades a little image fidelity for a large
#     drop in wire size, which is the whole point for high-rate video obs.
#   * Every other ndarray (float state vectors, action chunks, ...) is written
#     as its raw buffer plus dtype and shape, and rebuilt from exactly those --
#     a byte-exact, lossless round trip.
#   * A numpy scalar is written using its underlying Python value.
#
# Dtypes whose bytes would not round-trip cleanly (object, void, complex) are
# rejected up front.

_OPAQUE_KINDS = frozenset("OVc")  # object / void / complex
_DEFAULT_JPEG_QUALITY = 95


def _is_image(value) -> bool:
    """True for a uint8 array shaped like one or more channels-last images.

    Matches ``(..., H, W, C)`` with ``C in (1, 3)`` and at least the three
    trailing image axes, so a time-stacked ``(T, H, W, 3)`` obs and a lone
    ``(H, W, 3)`` frame both qualify while float state / action arrays never do.
    """
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.uint8
        and value.ndim >= 3
        and value.shape[-1] in (1, 3)
    )


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """Encode one ``(H, W, 3)`` or ``(H, W, 1)`` uint8 frame to JPEG bytes."""
    if frame.shape[-1] == 1:
        image = Image.fromarray(np.ascontiguousarray(frame[..., 0]), mode="L")
    else:
        image = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    buffer = io.BytesIO()
    # subsampling=0 keeps full chroma (4:4:4); paired with a high quality it
    # preserves the color detail that matters for control while still shrinking
    # the payload by an order of magnitude versus raw pixels.
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def _to_portable(value, *, use_jpeg: bool, quality: int):
    """``default`` hook: convert numpy values into msgpack-friendly maps."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in _OPAQUE_KINDS:
        raise ValueError(f"cannot serialize values of dtype {value.dtype!r}")
    if use_jpeg and _is_image(value):
        # Flatten every leading (batch / time) axis into a list of frames so an
        # arbitrarily stacked image collapses to one JPEG per frame.
        frames = value.reshape((-1,) + value.shape[-3:])
        return {
            b"__jpeg__": True,
            b"frames": [_encode_jpeg(frame, quality) for frame in frames],
            b"shape": value.shape,
        }
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _from_portable(obj):
    """``object_hook``: rebuild the numpy values encoded by :func:`_to_portable`."""
    if b"__jpeg__" in obj:
        # Decode each frame, restack, and restore the original leading shape.
        # A grayscale frame decodes to (H, W); the reshape re-adds its trailing
        # channel axis because the element count matches the stored shape.
        frames = [np.asarray(Image.open(io.BytesIO(blob))) for blob in obj[b"frames"]]
        return np.stack(frames, axis=0).reshape(tuple(obj[b"shape"]))
    if b"__ndarray__" in obj:
        # Copy off the msgpack buffer: it is an immutable ``bytes``, so a view
        # onto it would be read-only and crash any in-place post-processing the
        # caller does to the reconstructed array (e.g. clipping an action chunk).
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"]).copy()
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class _Channel:
    """msgpack codec for a single connection.

    ``Packer`` keeps internal state and is meant to be reused, so every
    connection holds its own; decoding is stateless and stays a function.
    Image frames are JPEG-compressed when ``use_jpeg`` is set (the default);
    decoding accepts JPEG-encoded frames regardless, so a JPEG sender and a
    raw sender interoperate in either direction.
    """

    def __init__(self, *, use_jpeg: bool = True, jpeg_quality: int = _DEFAULT_JPEG_QUALITY) -> None:
        self._packer = msgpack.Packer(
            default=lambda value: _to_portable(value, use_jpeg=use_jpeg, quality=jpeg_quality)
        )

    def freeze(self, obj) -> bytes:
        return self._packer.pack(obj)

    @staticmethod
    def thaw(blob):
        return msgpack.unpackb(blob, object_hook=_from_portable)


# --------------------------------------------------------------------------- #
#  Caller side                                                                 #
# --------------------------------------------------------------------------- #


class RemotePolicy:
    """A local-looking handle to a policy that actually lives over a socket.

    Building one blocks until the server answers and its opening metadata frame
    lands; from then on ``get_action`` and ``reset`` are normal method calls
    that simply happen to make a round trip on the wire.
    """

    _RECONNECT_GAP = 5  # seconds between connection attempts while waiting

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = self._resolve_uri(host, port)
        self._api_key = api_key
        self._channel = _Channel()
        self._socket, self._metadata = self._handshake()

    @staticmethod
    def _resolve_uri(host: str, port: int | None) -> str:
        # If the caller already gave a ws:// address, use it unchanged;
        # otherwise build one from the host and append the port when present.
        base = host if host.startswith("ws") else f"ws://{host}"
        return base if port is None else f"{base}:{port}"

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _handshake(self):
        # Keep retrying until the connection succeeds; the first frame the
        # server sends back is its metadata, which we hold onto.
        auth = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        log.info("connecting to policy server at %s", self._uri)
        while True:
            try:
                socket = _open_client(
                    self._uri,
                    additional_headers=auth,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                )
                return socket, self._channel.thaw(socket.recv())
            except ConnectionRefusedError:
                log.info("server not up yet; retrying in %ss", self._RECONNECT_GAP)
                time.sleep(self._RECONNECT_GAP)

    def _request(self, payload: dict):
        # One request out, one response back. A response delivered as text
        # rather than binary signals the handler failed, so turn it into a
        # raised exception locally.
        self._socket.send(self._channel.freeze(payload))
        reply = self._socket.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server raised an error:\n{reply}")
        return self._channel.thaw(reply)

    def get_action(self, obs: dict) -> dict:
        return self._request({"type": "get_action", "obs": obs})

    def reset(self):
        return self._request({"type": "policy_reset"})


# --------------------------------------------------------------------------- #
#  Hosting side                                                                #
# --------------------------------------------------------------------------- #


class PolicyService:
    """Publish one local policy on a websocket endpoint.

    Every client first receives the metadata map, msgpack-encoded. From there
    the client steers the policy by sending request maps stamped with a
    ``type`` field; the matching action runs and its result goes straight back.
    If a handler raises, the client is handed the traceback as a plain text
    frame and the connection is closed with an error status.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        # Begin with the optional metadata argument and layer the policy's own
        # metadata on top, so the policy wins wherever the two overlap.
        self._metadata = {**(metadata or {}), **policy.metadata}
        # request "type" string -> handler callable
        self._routes = {
            "get_action": lambda req: self._policy.get_action(req["obs"]),
            "policy_reset": lambda req: self._policy.reset(),
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def run_forever(self) -> None:
        """Serve clients on the calling thread until interrupted."""
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        async with _open_server(
            self._on_connection,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # Keepalive pings are turned off on both sides. A single request
            # can keep the handler busy longer than the default ping interval,
            # and we don't want that slow turnaround mistaken for a dead peer.
            ping_interval=None,
            ping_timeout=None,
            process_request=_liveness_probe,
        ) as server:
            await server.serve_forever()

    async def _on_connection(self, socket) -> None:
        who = socket.remote_address
        log.info("client %s connected", who)
        channel = _Channel()

        # Send the metadata map before handling any requests.
        await socket.send(channel.freeze(self._metadata))

        try:
            async for frame in socket:
                request = channel.thaw(frame)
                kind = request.get("type")
                route = self._routes.get(kind)
                if route is None:
                    raise ValueError(f"unrecognized request type: {kind!r}")
                await socket.send(channel.freeze(route(request)))
        except ConnectionClosed:
            log.info("client %s disconnected", who)
        except Exception:
            # Hand the failure text back to the client so the caller can see
            # what happened, then close the connection and re-raise here.
            await socket.send(traceback.format_exc())
            await socket.close(
                code=CloseCode.INTERNAL_ERROR,
                reason="server-side failure; traceback in the previous frame",
            )
            raise


def _liveness_probe(connection, request):
    """Answer a plain HTTP GET on the liveness path; any other request
    continues into the normal websocket upgrade."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
