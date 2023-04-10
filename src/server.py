import argparse
import socket
import pickle
import struct
import torch
import torchvision.models as models


class Server:
    def __init__(self, model, host='127.0.0.1', port=12345):
        self.model = model
        self.host = host
        self.port = port

    def recvall(self, conn, size):
        data = bytearray()
        while len(data) < size:
            part = conn.recv(size - len(data))
            if not part:
                return None
            data.extend(part)
        return data

    def recv_object(self, conn):
        data_len_bytes = self.recvall(conn, 4)
        if not data_len_bytes:
            return None
        data_len = struct.unpack('!I', data_len_bytes)[0]
        data = self.recvall(conn, data_len)
        return pickle.loads(data)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            print(f"Server started at {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    while True:
                        input_tensor = self.recv_object(conn)
                        if input_tensor is None:
                            break

                        x = self.model.layer3(input_tensor)
                        x = self.model.layer4(x)
                        x = self.model.avgpool(x)
                        x = torch.flatten(x, 1)
                        output = self.model.fc(x)

                        output_data = pickle.dumps(output)
                        conn.sendall(struct.pack(
                            '!I', len(output_data)) + output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 Offloading Server')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address to bind the server to')
    parser.add_argument('--port', type=int, default=12345,
                        help='Port number to bind the server to')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Torchvision model to use')
    args = parser.parse_args()

    model = models.__dict__[args.model](weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    server = Server(model, host=args.ip, port=args.port)
    server.start()
