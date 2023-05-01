import json
import numpy as np
import matplotlib.pyplot as plt

N = 30
T = 0.1
NX = 8
NU = 2
NO = 3
dt = 1e-3

X_MIN   = -10.0
X_MAX   =  200.0
Y_MIN   = -4.0
Y_MAX   =  4.0
PSI_MIN = -np.inf
PSI_MAX =  np.inf
U_MIN   =  0.1
U_MAX   =  10.0
V_MIN   = -np.inf
V_MAX   =  np.inf
R_MIN   = -np.inf
R_MAX   =  np.inf
SA_MIN  = -np.inf
SA_MAX  =  np.inf
AX_MIN  = -5.0
AX_MAX  =  2.0
SR_MIN  = -5*np.pi/180
SR_MAX  =  5*np.pi/180
JX_MIN  = -5.0
JX_MAX  =  5.0
SM_OBS  =  1.25

lower_bounds = [X_MIN, Y_MIN, PSI_MIN, U_MIN, V_MIN, R_MIN, SA_MIN, AX_MIN, SR_MIN, JX_MIN]
upper_bounds = [X_MAX, Y_MAX, PSI_MAX, U_MAX, V_MAX, R_MAX, SA_MAX, AX_MAX, SR_MAX, JX_MAX]

def singleshoot_method():
    open_elapsed = []
    open_cost = []
    ipopt_elapsed = []
    ipopt_cost = []
    
    with open('./../method_singleshoot/open_singleshoot.json', 'r') as file:
        for line in file:
            run_data = json.loads(line)
            open_elapsed.append(run_data["elapsed"])
            open_cost.append(run_data["cost"])
            
    with open('./../nmpc_ipopt/ipopt_singleshoot.json', 'r') as file:
        for line in file:
            run_data = json.loads(line)
            ipopt_elapsed.append(run_data["elapsed"])
            ipopt_cost.append(run_data["cost"])

    x_axis = np.arange(1, 21)
    f, ax = plt.subplots(figsize=(17,3))
    f.subplots_adjust(left=0.05, bottom=0.15, right=0.98, top=0.95, wspace=None, hspace=None)
    ax.bar(x_axis - 0.25/2, open_cost, color='tab:blue', width=0.25, label='OpEn', edgecolor='black')
    ax.bar(x_axis + 0.25/2, ipopt_cost, color='tab:green', width=0.25, label='IPOPT', edgecolor='black')
    ax.set_xticks(x_axis)
    ax.set_xlabel('Case')
    ax.set_ylabel('Cost')
    ax.legend()
    
    f, ax = plt.subplots(figsize=(17,3))
    f.subplots_adjust(left=0.05, bottom=0.15, right=0.98, top=0.95, wspace=None, hspace=None)
    bp1 = ax.boxplot(open_elapsed, positions=x_axis, showfliers=False, widths=0.25, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot(ipopt_elapsed, positions=x_axis + 0.25, showfliers=False, widths=0.25, patch_artist=True, boxprops=dict(facecolor="C2"))
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['OpEn', 'IPOPT'], loc='upper right')
    ax.set_xticks(x_axis)
    ax.set_xlabel('Case')
    ax.set_ylabel('Solve time [s]')
    plt.show()

def main():
    singleshoot_method()

    
if __name__ == "__main__":
    main()