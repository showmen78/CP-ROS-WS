"""Shared TCP receiver for newline-separated OpenCDA JSON messages."""

import json
import queue
import socket
import threading


TCP_HOST = "127.0.0.1"


class TcpJsonReceiver:
    """Receive JSON on one TCP port and store it until a ROS node reads it."""

    def __init__(self, tcp_port, logger):
        self.tcp_port = tcp_port
        self.logger = logger
        self.messages = queue.Queue()
        self.stop_event = threading.Event()
        self.server_socket = None
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server.bind((TCP_HOST, self.tcp_port))
            server.listen(1)
            server.settimeout(1.0)
            self.server_socket = server
        except OSError as error:
            self.logger.error(
                "Could not open TCP port {}: {}".format(
                    self.tcp_port,
                    error,
                )
            )
            server.close()
            return

        self.logger.info(
            "Waiting for OpenCDA on {}:{}".format(
                TCP_HOST,
                self.tcp_port,
            )
        )

        while not self.stop_event.is_set():
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            connection.settimeout(1.0)
            receive_buffer = b""

            try:
                while not self.stop_event.is_set():
                    try:
                        received = connection.recv(4096)
                    except socket.timeout:
                        continue

                    if not received:
                        break

                    receive_buffer += received
                    while b"\n" in receive_buffer:
                        raw_message, receive_buffer = receive_buffer.split(
                            b"\n",
                            1,
                        )
                        if not raw_message.strip():
                            continue

                        try:
                            message = json.loads(raw_message.decode("utf-8"))
                            self.messages.put(message)
                        except (ValueError, UnicodeDecodeError) as error:
                            self.logger.warning(
                                "Invalid JSON received: {}".format(error)
                            )
            except (ConnectionResetError, OSError):
                pass
            finally:
                connection.close()

        server.close()

    def get_messages(self):
        """Return all messages currently waiting in the queue."""
        messages = []
        while True:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                return messages

    def close(self):
        """Stop the receiver thread and close the TCP socket."""
        self.stop_event.set()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
