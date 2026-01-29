import coincurve
import struct
import numpy as np
import os
import time

N_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def bytes_to_u64_le(b32_be: bytes):
    return list(struct.unpack("<4Q", b32_be[::-1]))

def scalar_to_point_u64(scalar: int):
    scalar %= N_ORDER
    if scalar == 0:
        return [0]*4, [0]*4
    pk = coincurve.PrivateKey.from_int(scalar)
    pt = pk.public_key.format(compressed=False) 
    x_be, y_be = pt[1:33], pt[33:65]
    return bytes_to_u64_le(x_be), bytes_to_u64_le(y_be)

def build_comb_table_u64(COMB_W=16, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"comb_table_w{COMB_W}.npy")
    if os.path.exists(cache_file):
        print(f"[{time.strftime('%H:%M:%S')}] Carregando tabela combinatória do cache: {cache_file}")
        start = time.time()
        table_flat = np.load(cache_file)
        print(f"[{time.strftime('%H:%M:%S')}] Tabela carregada em {time.time() - start:.2f}s | shape={table_flat.shape}")
        return table_flat
    print(f"[{time.strftime('%H:%M:%S')}] Gerando tabela combinatória (W={COMB_W})...")
    start = time.time()
    TABLE_SIZE = 1 << COMB_W
    table = np.zeros((TABLE_SIZE, 8), dtype=np.uint64)
    for i in range(TABLE_SIZE):
        scalar = 0
        for w in range(COMB_W):
            scalar |= ((i >> w) & 1) << (w * 16)
        x4, y4 = scalar_to_point_u64(scalar)
        table[i, 0:4] = x4
        table[i, 4:8] = y4
    table_flat = np.ascontiguousarray(table.reshape(-1))
    print(f"[{time.strftime('%H:%M:%S')}] Salvando tabela no cache: {cache_file}")
    np.save(cache_file, table_flat)
    print(f"[{time.strftime('%H:%M:%S')}] Tabela gerada e salva em {time.time() - start:.2f}s | shape={table_flat.shape}")
    return table_flat


