import torch


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m, self.new_m = torch.zeros(1), torch.zeros(1)
        self.old_s, self.new_s = torch.zeros(1), torch.zeros(1)

    def clear(self):
        self.n = 0

    def size(self):
        return self.n

    def shape(self):
        return self.new_m.shape

    def set_device(self, device):
        self.old_m, self.new_m = self.old_m.to(device), self.new_m.to(device)
        self.old_s, self.new_s = self.old_s.to(device), self.new_s.to(device)

    def push(self, x):
        self.n += 1

        self.set_device(x.device)

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = torch.zeros(1)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m.item() if self.n else torch.zeros(1).item()

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else torch.zeros(1).item()

    def std(self):
        return torch.sqrt(self.variance()).item()
