import threading
import socket
import time


"""
@brief  Host IP address. 0.0.0.0 referring to current 
        host/computer IP address.
"""
host_ip = "0.0.0.0"

"""
@brief  UDP port to receive response msg from Tello.
        Tello command response will send to this port.
"""
response_port = 8890

""" Welcome note """
print("\nTello Sensor States Program\n")


class Tello:
    def __init__(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host_ip, response_port))  # Bind for receiving

    def terminate(self):
        self._running = False
        self.sock.close()

    def recv(self):
        """ Handler for Tello states message """
        while self._running:
            try:
                msg, _ = self.sock.recvfrom(1024)  # Read 1024-bytes from UDP socket
                print("states: {}".format(msg.decode(encoding="utf-8")))
            except Exception as err:
                print(err)


""" Start new thread for receive Tello response message """
t = Tello()
recvThread = threading.Thread(target=t.recv)
recvThread.start()

while True:
    try:
        # Get input from CLI
        msg = input()

        # Check for "end"
        if msg == "bye":
            t.terminate()
            recvThread.join()
            print("\nGood Bye\n")
            break
    except KeyboardInterrupt:
        t.terminate()
        recvThread.join()
        break

