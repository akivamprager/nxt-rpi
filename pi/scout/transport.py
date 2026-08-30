"""Byte transports to the NXT.

Everything above this module works on an abstract byte stream, so the link can
be swapped without touching the protocol, robot, or mission layers.

Why Bluetooth is the default, and when to switch
------------------------------------------------
The Pi 3B's WiFi and Bluetooth are the same chip (BCM43438) sharing one
antenna in full time-division mode. Saturating WiFi with video while running
RFCOMM to the NXT causes Bluetooth stutter. The primary mitigation is
architectural — keep video off the WiFi hot path — but if telemetry still
stutters under load, the escalation path is:

1. Reduce camera frame rate and resolution.
2. Add an MT7610U/MT7612U USB dongle and move WiFi to 5 GHz.
3. Only then swap ``BluetoothTransport`` for ``UsbTransport``.

Watch ``FrameDecoder.checksum_errors`` on the dashboard: a steadily climbing
count is the signature of this problem.
"""

from __future__ import annotations

import abc
import errno
import os
import select
import socket
import time


class TransportError(IOError):
    """The link failed in a way that requires reopening it."""


class Transport(abc.ABC):
    """A bidirectional byte stream to the robot."""

    @abc.abstractmethod
    def open(self) -> None:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    def write(self, data: bytes) -> None:
        ...

    @abc.abstractmethod
    def read(self, max_bytes: int = 4096, timeout: float = 0.1) -> bytes:
        """Read up to ``max_bytes``.

        Returns ``b""`` if nothing arrived within ``timeout`` — a timeout is
        normal and not an error. Raises :class:`TransportError` if the link
        has actually dropped.
        """

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        ...

    def __enter__(self) -> "Transport":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _FdTransport(Transport):
    """Shared implementation for transports backed by a file descriptor."""

    def __init__(self) -> None:
        self._fd: int | None = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass  # Already gone; closing is best-effort by design.
            self._fd = None

    def write(self, data: bytes) -> None:
        if self._fd is None:
            raise TransportError("write on a closed transport")
        view = memoryview(data)
        while view:
            try:
                written = os.write(self._fd, view)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise TransportError(f"write failed: {exc}") from exc
            if written == 0:
                raise TransportError("write returned 0; the link is gone")
            view = view[written:]

    def read(self, max_bytes: int = 4096, timeout: float = 0.1) -> bytes:
        if self._fd is None:
            raise TransportError("read on a closed transport")
        try:
            ready, _, _ = select.select([self._fd], [], [], timeout)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                return b""
            raise TransportError(f"select failed: {exc}") from exc
        if not ready:
            return b""
        try:
            chunk = os.read(self._fd, max_bytes)
        except OSError as exc:
            if exc.errno in (errno.EINTR, errno.EAGAIN):
                return b""
            raise TransportError(f"read failed: {exc}") from exc
        if chunk == b"":
            # select said readable but nothing came back: EOF.
            raise TransportError("connection closed by the remote end")
        return chunk


class BluetoothTransport(_FdTransport):
    """RFCOMM link over ``/dev/rfcomm0``.

    Setup on the Pi, once::

        bluetoothctl              # scan on / pair <ADDR>  (PIN 1234) / trust
        sudo rfcomm bind 0 <ADDR> 1
        sudo usermod -aG dialout $USER   # then log out and back in

    Note BlueZ has deprecated the ``rfcomm`` tool. The bind does not survive a
    reboot on its own, so either add a systemd unit or call
    :meth:`wait_for_device` and re-bind from a helper script.
    """

    def __init__(self, device: str = "/dev/rfcomm0") -> None:
        super().__init__()
        self.device = device

    def open(self) -> None:
        if self._fd is not None:
            return
        try:
            # O_NOCTTY: this is a data link, not our controlling terminal.
            self._fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY)
        except FileNotFoundError as exc:
            raise TransportError(
                f"{self.device} does not exist. Bind it first:\n"
                f"    sudo rfcomm bind 0 <NXT_BT_ADDRESS> 1"
            ) from exc
        except PermissionError as exc:
            raise TransportError(
                f"no permission to open {self.device}. Add yourself to the "
                f"dialout group:\n    sudo usermod -aG dialout $USER\n"
                f"then log out and back in."
            ) from exc
        except OSError as exc:
            raise TransportError(
                f"could not open {self.device}: {exc}. Is the NXT switched on, "
                f"paired, and running ScoutServer?"
            ) from exc

    def wait_for_device(self, timeout: float = 30.0) -> bool:
        """Block until the rfcomm node appears, for use at boot."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(self.device):
                return True
            time.sleep(0.5)
        return False


class SocketTransport(_FdTransport):
    """Link over an existing socket.

    Used by the firmware simulator so the whole Pi stack can be exercised on a
    laptop with no NXT attached, and usable for a TCP bridge when debugging.
    """

    def __init__(self, sock: socket.socket) -> None:
        super().__init__()
        self._sock = sock

    def open(self) -> None:
        if self._fd is None:
            self._sock.setblocking(False)
            self._fd = self._sock.fileno()

    def close(self) -> None:
        # The fd belongs to the socket, so let the socket close it rather than
        # closing the raw fd out from under it.
        self._fd = None
        try:
            self._sock.close()
        except OSError:
            pass


class UsbTransport(Transport):
    """USB fallback. Not implemented — see the note below before starting.

    This exists as a documented escalation path, not as working code. Only
    reach for it if reducing the video rate and moving WiFi to 5 GHz both fail
    to stop Bluetooth dropouts.

    What it takes:

    - ``pyusb``; the NXT is VID ``0x0694``, PID ``0x0002``.
    - Bulk endpoints ``0x01`` (OUT) and ``0x82`` (IN) on interface 0, config 1.
    - Call ``dev.detach_kernel_driver(0)`` on Linux, or the interface stays claimed.
    - Use generous timeouts. The brick sits in ``USB.waitForConnection()`` and
      will not answer until the host sends the first packet.
    - Keep ``NXTConnection.RAW`` on the firmware side. leJOS's PACKET mode adds
      a 2-byte length header whose endianness is documented inconsistently
      between the Bluetooth and USB stacks; our own framing sidesteps that.
    - ``nxt-python`` cannot help here: it speaks LCP, which leJOS firmware does
      not answer.

    The firmware needs a matching change: swap ``Bluetooth.waitForConnection``
    for ``USB.waitForConnection(0, NXTConnection.RAW)`` in ``ScoutServer``.
    Everything above the transport stays as it is.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "UsbTransport is a documented fallback, not an implementation. "
            "See the class docstring, and exhaust the cheaper mitigations first."
        )

    def open(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def write(self, data: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def read(self, max_bytes: int = 4096, timeout: float = 0.1) -> bytes:  # pragma: no cover
        raise NotImplementedError

    @property
    def is_open(self) -> bool:  # pragma: no cover
        return False
