import os
import cv2
import math
import random
import time

# 1. Tensor Class
class Tensor:
    def __init__(self, data, shape=None):
        self.data = data
        self.grad = None
        self.shape = shape if shape else self._get_shape(data)

    def _get_shape(self, data):
        shape = []
        d = data
        while isinstance(d, list):
            shape.append(len(d))
            d = d[0] if len(d) > 0 else None
        return tuple(shape)

    def zero_grad(self):
        self.grad = self._zeros_like(self.data)

    def _zeros_like(self, data):
        if not isinstance(data, list): return 0.0
        return [self._zeros_like(d) for d in data]

# Initialize weights with He Initialization
def he_init(fan_in, shape):
    std = math.sqrt(2.0 / fan_in)
    def create_recursive(dims):
        if len(dims) == 1:
            return [random.gauss(0, std) for _ in range(dims[0])]
        return [create_recursive(dims[1:]) for _ in range(dims[0])]
    return create_recursive(shape)

# Flattening a 4D tensor
def flatten_tensor(data):
    batch_size = len(data)
    flattened = []
    for b in range(batch_size):
        flat_b = []
        for c in range(len(data[0])):
            for h in range(len(data[0][0])):
                flat_b.extend(data[b][c][h])
        flattened.append(flat_b)
    return flattened

# 2. Layers

class Layer:
    def forward(self, x): pass
    def backward(self, grad_output, learning_rate): pass
    # FIX 1: Accept *args so it doesn't crash if extra args are passed
    def get_params(self, *args): return 0, 0, 0 

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = [[random.gauss(0, math.sqrt(2.0/in_features)) for _ in range(out_features)] 
                  for _ in range(in_features)]
        self.b = [0.0] * out_features
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x
        batch_size = len(x)
        output = [[0.0] * self.out_features for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(self.out_features):
                s = self.b[j]
                for k in range(self.in_features):
                    s += x[i][k] * self.W[k][j]
                output[i][j] = s
        return output

    def backward(self, grad_output, lr):
        batch_size = len(grad_output)
        grad_W = [[0.0] * self.out_features for _ in range(self.in_features)]
        grad_b = [0.0] * self.out_features
        grad_input = [[0.0] * self.in_features for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(self.out_features):
                g = grad_output[i][j]
                grad_b[j] += g
                for k in range(self.in_features):
                    grad_W[k][j] += self.input_cache[i][k] * g
                    grad_input[i][k] += self.W[k][j] * g

        # Update weights
        for k in range(self.in_features):
            for j in range(self.out_features):
                self.W[k][j] -= lr * (grad_W[k][j] / batch_size)
        for j in range(self.out_features):
            self.b[j] -= lr * (grad_b[j] / batch_size)

        return grad_input

    # FIX 2: Accept *args to ignore H/W arguments from training loop
    def get_params(self, *args):
        params = (self.in_features + 1) * self.out_features
        macs = self.in_features * self.out_features
        flops = 2 * macs
        return params, macs, flops


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_c = in_channels
        self.out_c = out_channels
        self.k = kernel_size
        self.stride = stride
        self.pad = padding
        self.filters = he_init(in_channels * kernel_size**2, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = [0.0] * out_channels
        self.cache = None

    def forward(self, x):
        self.cache = x
        batch, in_c, h, w = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        out_h = (h - self.k + 2*self.pad) // self.stride + 1
        out_w = (w - self.k + 2*self.pad) // self.stride + 1
        
        output = [[[[0.0] * out_w for _ in range(out_h)] for _ in range(self.out_c)] for _ in range(batch)]
        
        for b in range(batch):
            for oc in range(self.out_c):
                res_bias = self.bias[oc]
                for oh in range(out_h):
                    for ow in range(out_w):
                        res = res_bias
                        h_start = oh * self.stride - self.pad
                        w_start = ow * self.stride - self.pad
                        for ic in range(in_c):
                            for kh in range(self.k):
                                for kw in range(self.k):
                                    in_h = h_start + kh
                                    in_w = w_start + kw
                                    if 0 <= in_h < h and 0 <= in_w < w:
                                        res += x[b][ic][in_h][in_w] * self.filters[oc][ic][kh][kw]
                        output[b][oc][oh][ow] = res
        return output

    def backward(self, grad_output, lr):
        x = self.cache
        batch, in_c, h, w = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        out_h, out_w = len(grad_output[0][0]), len(grad_output[0][0][0])
        
        # FIX 3: Initialize gradients for input to enable backprop
        grad_input = [[[[0.0]*w for _ in range(h)] for _ in range(in_c)] for _ in range(batch)]
        grad_filters = [[[[0.0]*self.k for _ in range(self.k)] for _ in range(self.in_c)] for _ in range(self.out_c)]
        grad_bias = [0.0] * self.out_c
        
        for b in range(batch):
            for oc in range(self.out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        g = grad_output[b][oc][oh][ow]
                        grad_bias[oc] += g
                        
                        h_start = oh * self.stride - self.pad
                        w_start = ow * self.stride - self.pad
                        
                        for ic in range(self.in_c):
                            for kh in range(self.k):
                                for kw in range(self.k):
                                    in_h = h_start + kh
                                    in_w = w_start + kw
                                    
                                    if 0 <= in_h < h and 0 <= in_w < w:
                                        # Gradient for Weights
                                        grad_filters[oc][ic][kh][kw] += x[b][ic][in_h][in_w] * g
                                        # FIX 4: Gradient for Input (Critical!)
                                        grad_input[b][ic][in_h][in_w] += self.filters[oc][ic][kh][kw] * g
        
        # Update Weights
        for oc in range(self.out_c):
            self.bias[oc] -= lr * (grad_bias[oc] / batch)
            for ic in range(self.in_c):
                for kh in range(self.k):
                    for kw in range(self.k):
                        self.filters[oc][ic][kh][kw] -= lr * (grad_filters[oc][ic][kh][kw] / batch)
        
        # FIX 5: Return the gradient!
        return grad_input

    # FIX 6: Accept arguments for spatial calculation
    def get_params(self, input_h=32, input_w=32):
        out_h = (input_h + 2*self.pad - self.k) // self.stride + 1
        out_w = (input_w + 2*self.pad - self.k) // self.stride + 1
        
        params = (self.k * self.k * self.in_c * self.out_c) + self.out_c # + Bias
        macs = (self.k * self.k * self.in_c * self.out_c) * (out_h * out_w)
        flops = 2 * macs
        return params, macs, flops


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.k = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        self.cache = x
        batch, c, h, w = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        out_h, out_w = h // self.stride, w // self.stride
        output = [[[[0.0] * out_w for _ in range(out_h)] for _ in range(c)] for _ in range(batch)]
        
        for b in range(batch):
            for ch in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i*self.stride, j*self.stride
                        max_val = -float('inf')
                        for ki in range(self.k):
                            for kj in range(self.k):
                                val = x[b][ch][h_start+ki][w_start+kj]
                                if val > max_val: max_val = val
                        output[b][ch][i][j] = max_val
        return output

    def backward(self, grad_output, lr):
        x = self.cache
        batch, c, h, w = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        grad_input = [[[[0.0]*w for _ in range(h)] for _ in range(c)] for _ in range(batch)]
        out_h, out_w = len(grad_output[0][0]), len(grad_output[0][0][0])

        for b in range(batch):
            for ch in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        g = grad_output[b][ch][i][j]
                        h_start, w_start = i*self.stride, j*self.stride
                        max_val = -float('inf')
                        max_idx = (0, 0)
                        for ki in range(self.k):
                            for kj in range(self.k):
                                val = x[b][ch][h_start+ki][w_start+kj]
                                if val > max_val:
                                    max_val = val
                                    max_idx = (h_start+ki, w_start+kj)
                        grad_input[b][ch][max_idx[0]][max_idx[1]] += g
        return grad_input


class CrossEntropyLoss:
    def forward(self, logits, labels):
        batch_size = len(logits)
        self.probs = []
        loss = 0.0
        for i in range(batch_size):
            row = logits[i]
            max_val = max(row)
            exps = [math.exp(v - max_val) for v in row]
            sum_exps = sum(exps)
            softmax = [e / sum_exps for e in exps]
            self.probs.append(softmax)
            correct_class_prob = softmax[labels[i]]
            loss -= math.log(correct_class_prob + 1e-9)
        return loss / batch_size

    def backward(self, labels):
        batch_size = len(self.probs)
        grad = []
        for i in range(batch_size):
            row_grad = list(self.probs[i])
            row_grad[labels[i]] -= 1.0
            grad.append([g / batch_size for g in row_grad])
        return grad


def load_dataset(parent_path):
    print(f"Loading dataset from {parent_path}...")
    start_time = time.time()
    X, y = [], []
    classes = sorted(os.listdir(parent_path))
    class_map = {name: i for i, name in enumerate(classes)}
    
    for cls_name in classes:
        cls_folder = os.path.join(parent_path, cls_name)
        if not os.path.isdir(cls_folder): continue
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (32, 32))
                img_data = [[float(px)/255.0 for px in row] for row in img]
                X.append([img_data])
                y.append(class_map[cls_name])
                
    loading_time = time.time() - start_time
    print(f"Dataset loaded. Images: {len(X)}. Time: {loading_time:.4f}s")
    return X, y, loading_time

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        if isinstance(x[0][0], list): # 4D
             return [[[[max(0, v) for v in row] for row in ch] for ch in b] for b in x]
        else: # 2D
             return [[max(0, v) for v in row] for row in x]

    def backward(self, grad_output, lr):
        x = self.input
        if isinstance(x[0][0], list): # 4D
            return [[[[g if v > 0 else 0 for v, g in zip(r_in, r_out)] 
                      for r_in, r_out in zip(c_in, c_out)] 
                      for c_in, c_out in zip(b_in, b_out)] 
                      for b_in, b_out in zip(x, grad_output)]
        else: # 2D
            return [[g if v > 0 else 0 for v, g in zip(r_in, r_out)] 
                    for r_in, r_out in zip(x, grad_output)]

def get_batch(X, y, batch_size):
    indices = list(range(len(X)))
    random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i+batch_size]
        yield [X[k] for k in batch_idx], [y[k] for k in batch_idx]