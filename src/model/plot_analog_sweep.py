import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

csv_path = "mnist_analog_sweep.csv"

df = pd.read_csv(csv_path)

print(df.head())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = df["forward_out_noise"].values
y = df["backward_out_noise"].values
z = df["test_acc"].values
c = df["desired_bl"].values

sc = ax.scatter(x, y, z, c=c, cmap="viridis", marker="o")

ax.set_xlabel("forward_out_noise")
ax.set_ylabel("backward_out_noise")
ax.set_zlabel("test_acc")

cb = fig.colorbar(sc, ax=ax)
cb.set_label("desired_bl")

plt.title("Test accuracy vs analog noise and pulse length")
plt.tight_layout()
plt.show()
