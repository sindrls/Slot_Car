import socket


class CarreraUDPSender:
    '''
    Usage:

    sender = CarreraUDPSender()
    sender.send(80)

    0 is the lowest speed, 255 is the highest

    '''
    def __init__(self, ip: str = "192.168.97.61"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, 12021)  # Make sure the IP is correct

    def send(self, msg: int):  # msg is in the 0-255 range
        b_msg = msg.to_bytes(1, 'little')
        self.sock.sendto(b_msg, self.addr)