
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['DISPLAY'] = 'localhost:10.0'

import sys 
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
import pygame
import socket
import pickle
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import time
import threading
import torchvision.models as models
from PIL import Image
from pygame.locals import *
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize
)

from src.monitor import (
    get_sensor_data_orange,
    get_sensor_data_jetson, 
    get_sensor_data_demo,
    update_dataframe, 
    clear_terminal
)

class Client:
    def __init__(self, model, host='127.0.0.1', port=12345, filename='../data/constraints.json', demo=False, delay=0):
        self.model = model
        self.host = host
        self.port = port
        self.demo = demo
        self.delay = delay

        self.text = ""
        self.predicted_class = ""
        self.sensor_data = np.array([0, 0, 0, 0, 0, 0, 0])
        self.render_mode = "human"

        with open(filename, 'r') as f:
            self.constraints = json.load(f)

        self.render_mode = "human"

        self.window = None
        self.clock = None

        self.labels = [
            "AUX", "CPU", "thermal", "Tboard", "AO", "GPU", "Tdiode"
        ]

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
    
    def sensors(self):
        if self.demo:
            return get_sensor_data_demo()
        else:
            return get_sensor_data_jetson()
    
    def _render_frame(self):
        text = self.text
        predicted_class = self.predicted_class
        sensor_data = self.sensor_data

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((1366, 720))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.set_caption("Constraint Violation")
        font = pygame.font.Font(None, 24)

        canvas = pygame.Surface((1366, 720))
        canvas.fill((0, 0, 0))

        # Render the text message
        y = 20
        for line in text.split('\n'):
            text_surface = font.render(line.strip().replace('\t', '    '), True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.topleft = (20, y)
            canvas.blit(text_surface, text_rect)
            y += 30

        # Render the predicted class
        predicted_text = f"Predicted class: {predicted_class}"
        predicted_surface = font.render(predicted_text, True, (255, 255, 255))
        predicted_rect = predicted_surface.get_rect()
        predicted_rect.topleft = (20, y)
        canvas.blit(predicted_surface, predicted_rect)

        # Create a gradient bar plot for the sensor data
        n_bars = len(sensor_data)
        colormap = matplotlib.colormaps["inferno"]
        norm = plt.Normalize(sensor_data.min(), sensor_data.max())
        colors = colormap(norm(sensor_data))

        bar_width = 50  # Thicker bars
        max_bar_height = 300
        spacing = 15
        bars_right_margin = 100

        for i, color in enumerate(colors):
            bar_height = (sensor_data[i] - sensor_data.min()) / (sensor_data.max() - sensor_data.min()) * max_bar_height
            bar_x = 1366 - bars_right_margin - (n_bars - i) * (bar_width + spacing)
            pygame.draw.rect(canvas, (color[:3] * 255), (bar_x, 150 + (max_bar_height - bar_height), bar_width, bar_height))

            # Render the temperature
            temperature_text = f"{sensor_data[i]:.1f}"
            temperature_surface = font.render(temperature_text, True, (255, 255, 255))
            temperature_rect = temperature_surface.get_rect()
            temperature_rect.center = (bar_x + (bar_width / 2), 150 + max_bar_height + 10)
            canvas.blit(temperature_surface, temperature_rect)

        for i, label in enumerate(self.labels):
            label_surface = font.render(label, True, (255, 255, 255))
            label_rect = label_surface.get_rect()
            label_rect.center = (1366 - bars_right_margin - (n_bars - i) * (bar_width + spacing) + (bar_width / 2), 150 - 30)
            canvas.blit(label_surface, label_rect)
            
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(20)
    
    def predict(self, x):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            
            for i, layer in enumerate(self.model.children()):
                if np.average(np.array(self.sensors())) > self.constraints.get("avg") or \
                    np.max(np.array(self.sensors())) > self.constraints.get("max"):
                    self.send_object(s, {"data": x, "layer": i})
                    output = self.recv_object(s)
                    _, predicted = torch.max(output.data, 1)
                    predicted_class = idx_to_label[predicted.item()]

                    self.text = f"Constraints were violated at layer {i}.\nConstraint:\n\tavg: {round(self.constraints.get('avg'))}, max: {round(self.constraints.get('max'))}.\nViolation:\n\tavg: {round(np.average(np.array(self.sensors())))}, max: {round(np.max(np.array(self.sensors())))}.\nSending data to server..."
                    self.sensor_data = np.array(self.sensors())
                    self.predicted_class = predicted_class
                    self._render_frame()

                    return output
                else:
                    x = layer(x)
            return x
        
    def process_dataset(self, dataset, input_transforms, idx_to_label):
        self.model = self.model.eval()
        for filename in os.listdir(dataset):
            if not is_valid_image_path(os.path.join(dataset, filename)):
                continue
            
            try:
                # Load the image
                image = Image.open(os.path.join(dataset, filename))
                image = input_transforms(image)
                image = image.unsqueeze(0)

                # Get the output
                output = self.predict(image)
                _, predicted = torch.max(output.data, 1)
                print('Predicted: {}'.format(idx_to_label[predicted.item()]))
            except Exception as e:
                continue

            if self.delay > 0:
                time.sleep(self.delay)
        
    def process_random(self, duration=10):
        start_time = time.time()

        while time.time() - start_time < duration:
            # Generate a random image
            random_image = torch.rand(1, 3, 224, 224)

            # Process the random image using a thread
            thread = threading.Thread(
                target=self.predict, args=(random_image,))
            thread.start()
            thread.join()


def load_imagenet_classes(txt_file='../data/imagenet.txt'):
    with open(txt_file) as f:
        labels = [line.strip() for line in f.readlines()]
    idx_to_label = {idx: label for idx, label in enumerate(labels)}
    return idx_to_label


def is_valid_image_path(path):
    try:
        # Check if the path exists
        if not os.path.exists(path):
            return False

        # Check if the path is a file
        if not os.path.isfile(path):
            return False

        # Check if the file has an image extension (this is not foolproof)
        valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
        _, extension = os.path.splitext(path)
        if extension.lower() not in valid_image_extensions:
            return False

        # Try to open the file with PIL to check if it's a valid image
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False
    except Exception as e:
        print(e)
        return False
    

def is_valid_image_dir(path):
    try:
        # Check if the path exists
        if not os.path.exists(path):
            return False

        # Check if the path is a directory
        if not os.path.isdir(path):
            return False

        # Check if the directory contains at least one valid image
        for idx, filename in enumerate(os.listdir(path)):
            if not is_valid_image_path(os.path.join(path, filename)):
                print('Invalid image: {}'.format(filename))
                return False
            # else: 
            #     print(f'[{idx}] Valid image: {filename}')
                
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 Offloading Client')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of the server')
    parser.add_argument('--port', type=int, default=12345, help='Port number of the server')
    parser.add_argument('--model', type=str, default='resnet18', help='Torchvision model to use')
    parser.add_argument('--image', type=str, default=None, help='Path to the input image')
    parser.add_argument('--duration', type=int, default=None, help='Duration of the random image processing')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset directory')
    parser.add_argument('--filename', type=str, default='../config/constraints.json', help='Path to the constraints file')
    parser.add_argument('--labels', type=str, default='../data/imagenet.txt', help='Path to the labels file')
    parser.add_argument('--demo', action='store_true', help='Run the demo')
    parser.add_argument('--delay', type=int, default=0, help='Delay between each image processing')
    args = parser.parse_args()

    model = models.__dict__[args.model](weights="IMAGENET1K_V1")
    model.eval()

    idx_to_label = load_imagenet_classes(args.labels)

    input_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image == 'random':
        client = Client(model, host=args.ip, port=args.port, filename=args.filename, demo=args.demo, delay=args.delay)
        client.process_random(duration=args.duration)
    if is_valid_image_path(args.image):
        img = Image.open(args.image).convert('RGB')
        input_data = input_transforms(img).unsqueeze(0)
        client = Client(model, host=args.ip, port=args.port, filename=args.filename, demo=args.demo, delay=args.delay)
        output = client.predict(input_data)
        _, predicted = torch.max(output, 1)
        predicted_label = idx_to_label[predicted.item()]
        print('Predicted class:', predicted.item(), '-', predicted_label)
    if is_valid_image_dir(args.dataset):
        client = Client(model, host=args.ip, port=args.port, filename=args.filename, demo=args.demo, delay=args.delay)
        client.process_dataset(args.dataset, input_transforms, idx_to_label)