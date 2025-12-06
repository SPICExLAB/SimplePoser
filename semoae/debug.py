# Quick diagnostic - add this to your training script
import matplotlib.pyplot as plt
from semoae.dataset import SynPairedIMUData
# After loading data, before training:
train_dataset = SynPairedIMUData('/data/projects/Pose/dataset_work/IMUPoser/train.pt')

# Sample some data
for i in range(10):
    syn, real = train_dataset[i]
    print(f"Syn range: [{syn.min():.2f}, {syn.max():.2f}]")
    print(f"Real range: [{real.min():.2f}, {real.max():.2f}]")

    # Check the actual difference
    diff = (real - syn).numpy()
    print(f"Difference mean: {diff.mean():.4f}")
    print(f"Difference std: {diff.std():.4f}")

# Plot histogram of differences
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(diff[:15], bins=50, alpha=0.7, label='Acc diff')
plt.title('Acceleration differences')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(diff[15:], bins=50, alpha=0.7, label='Ori diff')
plt.title('Orientation differences')
plt.legend()

plt.savefig('data_difference_hist.png')
print("Saved data_difference_hist.png")