import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
from dysts.maps import Henon
from sklearn.preprocessing import StandardScaler

N_STEPS=100
TEST_FRAC=0.2

henon = Henon(a=1.4, b=0.3).make_trajectory(N_STEPS + 1 + 1000)[1000:]
n_test = int(N_STEPS * TEST_FRAC)
scaler = StandardScaler().fit(henon[: -n_test - 1])
henon_scaled = scaler.transform(henon)
u = henon_scaled[:-1]  # input at time t
y = henon_scaled[1:]  # target: Henon state at t+1

fig, ax = plt.subplots(dpi=144, figsize=(3, 1.5), constrained_layout=True)
ax.plot(u[:, 0], "k-")
ax.set(xlabel='time', ylabel="x(t)", xticks=[], yticks=[])
fig.savefig("plots/henon_timeseries.pdf", bbox_inches="tight")