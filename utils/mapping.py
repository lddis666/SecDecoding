

import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def build_mapping_gpu(tok_A,
                      tok_B,
                      even_share=True,
                      num_workers=8,
                      device="cuda",
                      dtype=torch.float32):
    """返回 COO 稀疏矩阵 M∈R^{|V_A|×|V_B|}（已在 GPU，列归一化）"""
    vocab_B_items = list(tok_B.get_vocab().items())
    vocab_size_A  = len(tok_A)
    vocab_size_B  = len(tok_B)

    rows, cols, data = [], [], []

    # ---------------- 多线程 decode/encode ----------------
    def process(chunk):
        out = []
        for s, id_B in chunk:
            txt   = tok_B.decode([id_B],
                                 skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
            ids_A = tok_A.encode(txt, add_special_tokens=False)
            if not ids_A:
                continue
            w = 1.0 / len(ids_A) if even_share else 1.0
            out.append((ids_A, id_B, w))
        return out

    chunk = (len(vocab_B_items) + num_workers - 1) // num_workers
    chunks = [vocab_B_items[i:i+chunk] for i in range(0, len(vocab_B_items), chunk)]

    with ThreadPoolExecutor(num_workers) as ex:
        for res in tqdm(ex.map(process, chunks), total=len(chunks), desc="build mapping"):
            for ids_A, id_B, w in res:
                rows.extend(ids_A)
                cols.extend([id_B] * len(ids_A))
                data.extend([w]    * len(ids_A))

    # ---------------- 创建 COO 稀疏 ----------------
    idx  = torch.tensor([rows, cols], dtype=torch.int64)
    vals = torch.tensor(data,        dtype=dtype)
    M = torch.sparse_coo_tensor(idx, vals,
                                size=(vocab_size_A, vocab_size_B),
                                dtype=dtype).coalesce()

    # ---------------- 列归一化 (仍保持稀疏) ----------
    col_sum   = torch.sparse.sum(M, dim=0).to_dense()          # (|V_B|)
    inv_col   = torch.where(col_sum > 0,
                            1.0 / col_sum,
                            torch.zeros_like(col_sum))
    new_vals  = M.values() * inv_col[M.indices()[1]]           # 逐元素缩放
    M = torch.sparse_coo_tensor(M.indices(), new_vals,
                                M.size(), dtype=dtype).coalesce().to(device)
    return M



# def build_mapping_gpu(tok_A,
#                       tok_B,
#                       even_share=True,
#                       num_workers=8,
#                       device="cuda",
#                       dtype=torch.float16):
#     """
#     返回稀疏矩阵 M ∈ R^{|V_A| × |V_B|}，列归一化，已放到 GPU。
#     完整遍历 0 … vocab_size_B-1，避免 get_vocab() 缺 id 问题。
#     """
#     vocab_size_A = tok_A.vocab_size           # = len(tok_A)
#     vocab_size_B = tok_B.vocab_size

#     # ---------- 0 … vocab_size_B-1 分块 ----------
#     all_ids_B = list(range(vocab_size_B))
#     chunk_size = (vocab_size_B + num_workers - 1) // num_workers
#     chunks = [all_ids_B[i:i+chunk_size] for i in range(0, vocab_size_B, chunk_size)]

#     rows, cols, data = [], [], []

#     def process(id_list):
#         out = []
#         for id_B in id_list:
#             # 把单个 id_B 解码成文本
#             txt = tok_B.decode([id_B],
#                                skip_special_tokens=False,
#                                clean_up_tokenization_spaces=False)
#             ids_A = tok_A.encode(txt, add_special_tokens=False)
#             if not ids_A:                 # 无法重编码，整列留空
#                 continue
#             w = 1.0 / len(ids_A) if even_share else 1.0
#             out.append((ids_A, id_B, w))
#         return out

#     with ThreadPoolExecutor(num_workers) as ex:
#         for res in tqdm(ex.map(process, chunks),
#                         total=len(chunks),
#                         desc="build mapping"):
#             for ids_A, id_B, w in res:
#                 rows.extend(ids_A)
#                 cols.extend([id_B]*len(ids_A))
#                 data.extend([w]*len(ids_A))

#     # ---------- 构造 COO 稀疏 ----------
#     idx  = torch.tensor([rows, cols], dtype=torch.int64)
#     vals = torch.tensor(data,           dtype=dtype)

#     M = torch.sparse_coo_tensor(idx, vals,
#                                 size=(vocab_size_A, vocab_size_B),
#                                 dtype=dtype).coalesce()

#     # ---------- 列归一化 ----------
#     col_sum = torch.sparse.sum(M, dim=0).to_dense()      # (|V_B|)
#     inv_col = torch.where(col_sum > 0,
#                           1.0 / col_sum,
#                           torch.zeros_like(col_sum))
#     new_vals = M.values() * inv_col[M.indices()[1]]
#     M = torch.sparse_coo_tensor(M.indices(), new_vals,
#                                 M.size(), dtype=dtype).coalesce()

#     return M.to(device)