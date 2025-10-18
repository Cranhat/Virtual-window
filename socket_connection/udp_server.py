import socket
import struct

class SocketServer:

    is_transfer_active = False

    def __init__(self, client_address, client_port = 5005):
        self.client_address = client_address
        self.client_port = client_port
    
    def __str__(self):
        return f"client_address: {self.client_address}, client_port: {self.client_port}"

    def create_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send(self, data): # tuple[float, float, float]
        data = bytes(data, "utf-8")
        self.sock.sendto(data, (self.client_address, self.client_port))

    def connect(self):
        self.sock.bind((self.client_address, self.client_port))
    
    def disconnect(self):
        self.sock.close()
        return 0
    
    def __enter__(self):
        self.create_socket()
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
                


    
    
    


    

