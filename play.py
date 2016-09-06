import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single
from network import BCPNN

np.set_printoptions(suppress=True)
sns.set(font_scale=2.0)

pattern1 = transform_normal_to_neural_single(np.array((1, 0, 0, 0, 0)))
pattern2 = transform_normal_to_neural_single(np.array((1, 0, 0, 0, 1)))
patterns = [pattern1, pattern2]

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

tau_z_post = 0.240
tau_z_pre = 0.240

nn = BCPNN(beta, w, p_pre=p, p_post=p, p_co=P, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre, g_a=1, M=2)

dt = 0.01
T = 10
time = np.arange(0, T + dt, dt)

history_o = np.zeros((time.size, beta.size))
history_s = np.zeros_like(history_o)
history_z_pre = np.zeros_like(history_o)
history_z_post = np.zeros_like(history_o)
history_a = np.zeros_like(history_o)

for index_t, t in enumerate(time):
    nn.update_continuous(dt)
    history_o[index_t, :] = nn.o
    history_s[index_t, :] = nn.s
    history_z_pre[index_t, :] = nn.z_pre
    history_z_post[index_t, :] = nn.z_post
    history_a[index_t, :] = nn.a

x = transform_neural_to_normal_single(nn.o)

# Plotting goes here
gs = gridspec.GridSpec(1, 2)
fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, history_o[:, 8], label='probability (o)')
ax1.plot(time, history_a[:, 8], label='adaptation')

ax1.set_xlabel('Time (ms)')
ax1.set_ylim([-0.1, 1.1])
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time, history_o[:, 9], label='probability (o)')
ax2.plot(time, history_a[:, 9], label='adaptation')

ax2.set_xlabel('Time (ms)')
ax2.set_ylim([-0.1, 1.1])
ax2.legend()

plt.show()
