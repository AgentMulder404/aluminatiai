import torch
import time

print("Starting 5-min GPU workload...")
a = torch.randn(8192, 8192, device='cuda')
b = torch.randn(8192, 8192, device='cuda')

end = time.time() + 300
i = 0
while time.time() < end:
    a = torch.nn.functional.relu(a @ b)
    b = torch.nn.functional.relu(b @ a)
    a = a / a.norm()
    b = b / b.norm()
    i += 1
    if i % 50 == 0:
        elapsed = int(300 - (end - time.time()))
        print(f"[{elapsed}s] step {i}")

print("Done.")
