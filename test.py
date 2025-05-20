import torch
print("CUDA tersedia:", torch.cuda.is_available())
print("Nama GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak terdeteksi")
