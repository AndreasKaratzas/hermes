import argparse
import socket
import pickle
import struct
import torch
import torchvision.models as models
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class Client:
    def __init__(self, model, host='127.0.0.1', port=12345):
        self.model = model
        self.host = host
        self.port = port

    def sendall(self, s, data):
        bytes_sent = 0
        while bytes_sent < len(data):
            sent = s.send(data[bytes_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            bytes_sent += sent

    def recvall(self, conn, size):
        data = bytearray()
        while len(data) < size:
            part = conn.recv(size - len(data))
            if not part:
                return None
            data.extend(part)
        return data

    def send_object(self, s, obj):
        data = pickle.dumps(obj)
        data_len = struct.pack('!I', len(data))
        self.sendall(s, data_len + data)

    def recv_object(self, conn):
        data_len_bytes = self.recvall(conn, 4)
        if not data_len_bytes:
            return None
        data_len = struct.unpack('!I', data_len_bytes)[0]
        data = self.recvall(conn, data_len)
        return pickle.loads(data)
    
    def predict(self, input_data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))

            x = self.model.conv1(input_data)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)

            self.send_object(s, x)
            output = self.recv_object(s)
            return output


def load_imagenet_classes(txt_file='imagenet.txt'):
    with open(txt_file) as f:
        labels = [line.strip() for line in f.readlines()]
    idx_to_label = {idx: label for idx, label in enumerate(labels)}
    return idx_to_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 Offloading Client')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of the server')
    parser.add_argument('--port', type=int, default=12345, help='Port number of the server')
    parser.add_argument('--model', type=str, default='resnet18', help='Torchvision model to use')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    model = models.__dict__[args.model](weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    idx_to_label = load_imagenet_classes()

    input_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert('RGB')
    input_data = input_transforms(img).unsqueeze(0)

    client = Client(model, host=args.ip, port=args.port)
    output = client.predict(input_data)
    _, predicted = torch.max(output, 1)
    predicted_label = idx_to_label[predicted.item()]
    print('Predicted class:', predicted.item(), '-', predicted_label)
