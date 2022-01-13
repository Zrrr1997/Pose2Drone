import threading
import socket
import time

"""
@brief  Tello IP address. Use local IP address since 
        host computer/device is a WiFi client to Tello.
"""
tello_ip = "192.168.10.1"

"""
Tello port to send command message.
"""
command_port = 8889

"""
@brief  Host IP address. 0.0.0.0 referring to current 
        host/computer IP address.
"""
host_ip = "0.0.0.0"

"""
@brief  UDP port to receive response msg from Tello.
        Tello command response will send to this port.
"""
response_port = 9000

""" Welcome note """
print("\nTello Command Program\n")


class Tello:
    def __init__(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host_ip, response_port))  # Bind for receiving

    def terminate(self):
        self._running = False
        self.sock.close()

    def recv(self):
        """ Handler for Tello response message """
        while self._running:
            try:
                msg, _ = self.sock.recvfrom(1024)  # Read 1024-bytes from UDP socket
                print("response: {}".format(msg.decode(encoding="utf-8")))
            except Exception as err:
                print(err)

    def send(self, msg):
        """ Handler for send message to Tello """
        msg = msg.encode(encoding="utf-8")
        self.sock.sendto(msg, (tello_ip, command_port))
        print("message: {}".format(msg))  # Print message



