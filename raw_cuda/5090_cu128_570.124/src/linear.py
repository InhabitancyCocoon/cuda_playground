import torch

M = 3
N = 4
K = 2

a = torch.arange(0, M * K).reshape(M, K)
b = torch.arange(0, K * N).reshape(K, N)
c = a @ b

d = torch.zeros_like(c)

for k in range(K):
    for i in range(M):
        for j in range(N):
            print(f"compute a[{i}][{k}] * b[{k}][{j}]")
            d[i][j] += a[i][k] * b[k][j]

e = torch.empty_like(c)

print("=" * 20)

for i in range(M):
    for j in range(N):
        acc = 0.
        for k in range(K):
            print(f"compute a[{i}][{k}] * b[{k}][{j}]")
            acc += a[i][k] * b[k][j]
        e[i][j] = acc

torch.testing.assert_close(d, c)
torch.testing.assert_close(d, e)


# a, b is row major

col_a = torch.empty_like(a).flatten()
col_b = torch.empty_like(b).flatten()

for i in range(M):
    for j in range(K):
        col_a[i + j * M] = a[i][j]

print(a)


for i in range(K):
    for j in range(N):
        col_b[i + j * K] = b[i][j]

print(b)

print(col_b.reshape(N, K))
print(col_a.reshape(K, M))
