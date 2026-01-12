import matplotlib.pyplot as plt

# -----------------------------
# Data (Seed42, item-cold p1)
# -----------------------------
ratios = [0.1, 0.3, 0.5]

# Warm NDCG@10
warm_mf = [0.210536, 0.248596, 0.375875]
warm_gcn = [0.183706, 0.257828, 0.328445]
warm_lgcn = [0.268890, 0.308331, 0.324054]

# Cold NDCG@100
cold_mf = [0.191023, 0.196035, 0.204329]
cold_gcn = [0.184890, 0.188182, 0.206737]
cold_lgcn = [0.207376, 0.204678, 0.225827]

# -----------------------------
# Plot 1: Warm NDCG@10
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(ratios, warm_mf, marker='o', label='MF')
plt.plot(ratios, warm_gcn, marker='s', label='GCN')
plt.plot(ratios, warm_lgcn, marker='^', label='LightGCN')

plt.xlabel("Item Cold Ratio")
plt.ylabel("Warm NDCG@10")
plt.title("Warm NDCG@10 vs Item Cold Ratio (Seed42)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("warm_ndcg_seed42.png", dpi=300)
plt.savefig("warm_ndcg_seed42.pdf")
plt.close()

# -----------------------------
# Plot 2: Cold NDCG@100
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(ratios, cold_mf, marker='o', label='MF')
plt.plot(ratios, cold_gcn, marker='s', label='GCN')
plt.plot(ratios, cold_lgcn, marker='^', label='LightGCN')

plt.xlabel("Item Cold Ratio")
plt.ylabel("Cold NDCG@100")
plt.title("Cold NDCG@100 vs Item Cold Ratio (Seed42)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cold_ndcg_seed42.png", dpi=300)
plt.savefig("cold_ndcg_seed42.pdf")
plt.close()

print("Saved:")
print(" - warm_ndcg_seed42.png / .pdf")
print(" - cold_ndcg_seed42.png / .pdf")