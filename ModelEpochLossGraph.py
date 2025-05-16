import matplotlib.pyplot as plt

epochs = list(range(1, 9))  # Epochs 1 to 8

same_race_losses = [0.3037, 0.2692, 0.2477, 0.2114, 0.1517, 0.0865, 0.0441, 0.0260]
diff_race_losses = [0.3110, 0.2750, 0.2505, 0.2300, 0.1802, 0.1204, 0.0655, 0.0413]  # Replace with actual values

plt.figure(figsize=(10, 6))
plt.plot(epochs, same_race_losses, marker='o', label='Same-Race Model')
plt.plot(epochs, diff_race_losses, marker='s', label='Diff-Race Model')

plt.title('Epoch Loss Comparison: Same-Race vs. Diff-Race Model')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.legend()
plt.xticks(epochs)
plt.tight_layout()
plt.show()

plt.savefig("epoch_loss_comparison.png")
