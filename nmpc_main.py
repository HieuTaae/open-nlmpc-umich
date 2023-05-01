import sys
import time
import json
import numpy as np
import casadi.casadi as cs
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "/Users/hewtaae/hieuta/courses-umich/2022-fall/mecheng-590/nmpc/method_singleshoot/nmpc_solver_debug/nlmpc")
# sys.path.insert(1, "/Users/hewtaae/hieuta/courses-umich/2022-fall/mecheng-590/nmpc/method_singleshoot/nmpc_solver_release/nlmpc")
import nlmpc
solver = nlmpc.solver()

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

def dyn_fn(x, u, solver_opt=1):
    # STATES
    X   = x[0]
    Y   = x[1]
    PSI = x[2]
    U   = x[3]
    V   = x[4]
    R   = x[5]
    SA  = x[6]
    Ax  = x[7]
    # CONTROLS
    SR  = u[0]
    Jx  = u[1]
    
    # VEHICLE PARAMETERS
    m    = 2.6887e+03
    Izz  = 4.1101e+03
    la   = 1.5775           # Distance from CoG to front axle
    lb   = 1.7245           # Distance form CoG to rear axle
    FzF0 = 1.3680e+04       # Static front axle load
    FzR0 = 1.2696e+04       # Static rear axle load
    
    # LOAD TRANSFER
    KZX  = 806.0            # Longitudinal load transfer coefficient
    KZYR = 1076.0           # Lateral load transfer coefficient rear axle
    KZYF = 675.0            # Lateral load transfer coefficient front axle
    
    # TIRE PARAMETERS
    FZ0  = 35000.0
    PCY1 = 1.5874           # Shape factor Cfy for lateral forces
    PDY1 = 0.73957          # Lateral friction Muy
    PDY2 = -0.075004        # Variation of friction with load
    PEY1 = 0.37562          # Lateral curvature Efy at Fznom
    PEY2 = -0.069325        # Varation of curvature Efy with load
    PEY3 = 0.29168          # Zero order camber dependency of curvature Efy
    PKY1 = -10.289          # Maximum value of stiffness Kfy/Fznom
    PKY2 = 3.3343           # Load at which Kfy reachs maximum value
    PHY1 = 0.0056509        # Horizontal shift Shy at Fznom
    PHY2 = -0.0020257       # Variation of shift Shy with load
    PVY1 = 0.015216         # Vertical shift in Svy/Fz at Fznom
    PVY2 = -0.010365        # Variation of shift Svy/Fz with load
    PC1  = PCY1
    PD1  = PDY1 - PDY2
    PD2  = PDY2 / FZ0
    PE1  = PEY1 - PEY2
    PE2  = PEY2 / FZ0
    PE3  = PEY3
    PK1  = PKY1 * FZ0
    PK2  = 1 / (PKY2 * FZ0)
    PH1  = PHY1 - PHY2
    PH2  = PHY2 / FZ0
    PV1  = PVY1 - PVY2
    PV2  = PVY2 / FZ0
    EP   = 0.01
        
    F_YF = 278.2789450759981 + ((PD2*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*(FzF0 - (Ax - V*R)*KZX))*cs.sin(PC1*cs.arctan((((PK1*cs.sin(2*cs.arctan(PK2*(FzF0 - (Ax - V*R)*KZX))))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)) + ((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX))**2 )**(0.5)))+EP))*((cs.arctan((V + la*R)/(U+EP)) - SA) + PH2*(FzF0 - (Ax - V*R)*KZX) + PH1)) - ((PE2*(FzF0 - (Ax - V*R)*KZX) + PE1)*(1 - PE3)*(((cs.arctan((V + la*R)/(U+EP)) - SA) + PH2*(FzF0 - (Ax - V*R)*KZX) + PH1))/((((cs.arctan((V + la*R)/(U+EP)) - SA) + PH2*(FzF0 - (Ax - V*R)*KZX) + PH1)**2 )**(0.5)))*((((PK1*cs.sin(2*cs.arctan(PK2*(FzF0 - (Ax - V*R)*KZX))))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)) + ((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX))**2 )**(0.5)))+EP))*((cs.arctan((V + la*R)/(U+EP)) - SA) + PH2*(FzF0 - (Ax - V*R)*KZX) + PH1)) - cs.arctan((((PK1*cs.sin(2*cs.arctan(PK2*(FzF0 - (Ax - V*R)*KZX))))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)) + ((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX)))/(((PD2*PC1*(FzF0 - (Ax - V*R)*KZX)**2 + PD1*PC1*(FzF0 - (Ax - V*R)*KZX))**2 )**(0.5)))+EP))*((cs.arctan((V + la*R)/(U+EP)) - SA) + PH2*(FzF0 - (Ax - V*R)*KZX) + PH1)))))) + (PV2*(FzF0 - (Ax - V*R)*KZX)**2 + PV1*(FzF0 - (Ax - V*R)*KZX)))
    
    F_YR = 259.95925391136984 + ((PD2*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*(FzR0 + (Ax - V*R)*KZX))*cs.sin(PC1*cs.arctan((((PK1*cs.sin(2*cs.arctan(PK2*(FzR0 + (Ax - V*R)*KZX))))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)) + ((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX))**2)**(0.5)))+EP))*((cs.arctan((V - lb*R)/(U+EP))) + PH2*(FzR0 + (Ax - V*R)*KZX) + PH1)) - ((PE2*(FzR0 + (Ax - V*R)*KZX) + PE1)*(1 - PE3*(((cs.arctan((V - lb*R)/(U+EP))) + PH2*(FzR0 + (Ax - V*R)*KZX) + PH1))/((((cs.arctan((V - lb*R)/(U+EP))) + PH2*(FzR0 + (Ax - V*R)*KZX) + PH1)**2 )**(0.5))))*((((PK1*cs.sin(2*cs.arctan(PK2*(FzR0 + (Ax - V*R)*KZX))))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)) + ((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX))**2)**(0.5)))+EP))*((cs.arctan((V - lb*R)/(U+EP))) + PH2*(FzR0 + (Ax - V*R)*KZX) + PH1)) - cs.arctan((((PK1*cs.sin(2*cs.arctan(PK2*(FzR0 + (Ax - V*R)*KZX))))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)) + ((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX)))/(((PD2*PC1*(FzR0 + (Ax - V*R)*KZX)**2 + PD1*PC1*(FzR0 + (Ax - V*R)*KZX))**2)**(0.5)))+EP))*((cs.arctan((V - lb*R)/(U+EP))) + PH2*(FzR0 + (Ax - V*R)*KZX) + PH1)))))) + (PV2*(FzR0 + (Ax - V*R)*KZX)**2 + PV1*(FzR0 + (Ax - V*R)*KZX)))
    
    FZ_RL = (0.5*(FzR0 + KZX*(Ax - V*R)) - KZYR*((F_YF + F_YR)/m)) # NOT USED IN THIS PROBLEM
    
    FZ_RR = (0.5*(FzR0 + KZX*(Ax - V*R)) + KZYR*((F_YF + F_YR)/m)) # NOT USED IN THIS PROBLEM
    
    if solver_opt == 1:
        dx = cs.SX.sym('dx', 8)
    else:
        dx = [0] * 8
    
    dx[0] = U*cs.cos(PSI) - (V + la*R)*cs.sin(PSI)    # X position
    dx[1] = U*cs.sin(PSI) + (V + la*R)*cs.cos(PSI)    # Y position
    dx[2] = R                                         # Yaw angle
    dx[3] = Ax                                        # Longitudinal speed
    dx[4] = (F_YF + F_YR)/m - R*U                     # Lateral speed
    dx[5] = (la*F_YF - lb*F_YR)/Izz                   # Yaw rate
    dx[6] = SR                                        # Steering angle
    dx[7] = Jx                                        # Longitudinal acceleration
    
    return dx

def goal_reached(states, goal=[110,0]):
    x_pos, y_pos = states[:2]
    x_goal, y_goal = goal
    xf_tol = 7.2
    
    distance = np.sqrt((x_pos - x_goal)**2 + (y_pos - y_goal)**2)
    
    if (distance <= xf_tol):
        return 1
    return 0

def visualize_trajectory(states, inputs, obstacle_list):
    def circle(obstacle, ax):
        angle = np.linspace(0, 2*np.pi, 150)
        obs_x = obstacle[0] + obstacle[2] * np.cos(angle)
        obs_y = obstacle[1] + obstacle[2] * np.sin(angle)
        obs_smx = obstacle[0] + (obstacle[2] + SM_OBS) * np.cos(angle)
        obs_smy = obstacle[1] + (obstacle[2] + SM_OBS) * np.sin(angle)
        ax.plot(obs_x, obs_y)
        ax.plot(obs_smx, obs_smy, '--r')
        
    global NX, NU, NO, SM_OBS, T, dt
    global lower_bounds, upper_bounds
    
    nx = len(states)
    nu = len(inputs)
    x   = [states[ii][0] for ii in range(nx)]
    y   = [states[ii][1] for ii in range(nx)]
    psi = [states[ii][2] for ii in range(nx)]
    u   = [states[ii][3] for ii in range(nx)]
    v   = [states[ii][4] for ii in range(nx)]
    w   = [states[ii][5] for ii in range(nx)]
    sa  = [states[ii][6] for ii in range(nx)]
    ax  = [states[ii][7] for ii in range(nx)]
    sr  = [inputs[ii][0] for ii in range(nu)]
    jx  = [inputs[ii][1] for ii in range(nu)]
    
    fig, axs = plt.subplots(figsize=(20, 3))
    axs.plot(x, y)
    circle(obstacle_list[0], axs)
    circle(obstacle_list[1], axs)
    circle(obstacle_list[2], axs)
    axs.set_xlim([0, 120])
    axs.set_ylim([-5, 5])
    
    tx = np.arange(nx) * T * dt
    tu = np.arange(nu) * T
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(tu, sr)
    axs[1].plot(tu, jx)
    
    plt.show()

def calculate_cost(states, inputs, obstacle_list):
    global NX, NU, NO, SM_OBS, T, dt
    global lower_bounds, upper_bounds
    
    J = 0
    nx = states.shape[0]
    nu = inputs.shape[0]
    tsim = int(T/dt)
    
    ## Cost to-go
    J += (10.0 * np.sum(inputs[:,0]**2 * tsim) + 0.01 * np.sum(inputs[:,1]**2 * tsim) + 0.1 * np.sum(states[:,1]**2)) * T
    ## Obstacle cost
    for ii in range(NO):
        x_obs, y_obs, r_obs = obstacle_list[ii]
        J += 60.0 * np.sum((np.tanh(-1.3*((states[:,0] - x_obs)**2/(r_obs + SM_OBS)**2 + (states[:,1] - y_obs)**2/(r_obs + SM_OBS)**2)) + 1)/2)
    ## Terminal cost
    J += 100 * np.sum(((states[np.arange(tsim, nx, tsim),0] - 110)**2 + (states[np.arange(tsim, nx, tsim),1] - 0)**2) / ((110 - states[np.arange(0, nx-tsim, tsim),0])**2 + (0 - states[np.arange(0, nx-tsim, tsim),1])**2) + 1)

    return J

def run_open(init_state, obstacle_list, guess_LagMul_attempt1, guess_LagMul_attempt2):
    global N, T, dt
    global NX, NU
    global SM_OBS
    global lower_bounds, upper_bounds
    
    num_iter = int(T/dt)
    guess_LagMul_attempt1 = [guess_LagMul_attempt1] * 330
    guess_LagMul_attempt2 = [guess_LagMul_attempt2] * 330
    obstacle_list_flat = [num for sublist in obstacle_list for num in sublist]
    
    states = [init_state]
    inputs = []
    elapsed_time = []
    
    guess = [0.0] * NU * N
    notconverge_cnt = 0
    total_iteration = 0
    success = 1
    iter = 0
    
    while not goal_reached(states[iter]):
        try:
            parameter = states[iter].copy()
            parameter.extend(obstacle_list_flat)
            
            ## Solver Input ##
            # print("Initial states:", parameter[:NX])
            # print("Sanity check:", states[-1])
            # print("GUESS:\n", guess)
            ##################
            
            tstart = time.time()
            ##
            result = solver.run(p=parameter, initial_guess=guess, initial_lagrange_multipliers=guess_LagMul_attempt1)
            if result is None:
                print("Running solution recovery.\n")
                tstart = time.time()
                result = solver.run(p=parameter, initial_guess=guess, initial_lagrange_multipliers=guess_LagMul_attempt2)
            tstop = time.time()
            elapsed = tstop - tstart
            
            ## Solver Output ##
            # print("Elapsed:", elapsed)
            # print("Exist status:", result.exit_status)
            # print("Outer iteration:", result.num_outer_iterations)
            # print("Inner iteration:", result.num_inner_iterations)
            # print("Last problem norm fpr:", result.last_problem_norm_fpr)
            # print("F1 infeasibility:", result.f1_infeasibility)
            # print("F2 norm:", result.f2_norm)
            # print("Penalty:", result.penalty)
            # print("Lagrangne multipliers:\n", result.lagrange_multipliers)
            # print("COST:", result.cost)
            # print("SOLUTION:\n", result.solution)
            # print("##############################")
            total_iteration += 1
            if result.exit_status != "Converged":
                notconverge_cnt += 1
            ###################
            
            u_opt = result.solution
            controls = u_opt[:NU]
            inputs.append(controls)
            elapsed_time.append(elapsed)
            
            for ii in range(num_iter):
                x_curr = states[iter]
                dx = dyn_fn(x_curr, controls, solver_opt=0)
                x_next = np.add(x_curr, np.multiply(dt, dx))
        
                states.append(x_next.tolist())
                iter += 1
            
            guess = u_opt
            guess_LagMul_attempt1 = result.lagrange_multipliers
        except:
            success = 0
            break
    
    return success, elapsed_time, states, inputs, total_iteration, notconverge_cnt

def main():
    init_states_list = [[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.0, 0.1], (2) [0.0, 0.1]
                        [0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.01, 0.3], (2) [0.0, 0.1]
                        [0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.01, 0.43], (2) [0.1, 0.4]
                        [0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.05, 0.55], (2) [0.05, 0.55]
                        [0.0, -2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.1, 0.4], (2) [0.01, 0.6]
                        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.05, 0.7], (2) [0.01, 0.7]
                        [0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.05, 0.55], (2) [0.1, 0.4]
                        [0.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.05, 0.5], (2) [0.05, 0.25]
                        [0.0, -1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], # (1) [0.01, 0.45], (2) [0.1, 0.4]
                        [0.0, -2.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]] # (1) [0.05, 0.55], (2) [0.08, 0.5]

    obstacle_list_list = [[[50.0, -1.0, 2.0],
                           [70.0,  1.0, 2.0],
                           [90.0, -2.0, 2.0]],
                          [[40.0, -1.0, 2.0],
                           [60.0,  1.0, 2.0],
                           [80.0, -2.0, 2.0]]]
    
    guess_LagMul_list = [[[0.0, 0.1],[0.01, 0.3],[0.01, 0.43],[0.05, 0.55],[0.1, 0.4],[0.05, 0.7],[0.05, 0.55],[0.05, 0.5],[0.01, 0.45],[0.05, 0.55]], 
                    [[0.0, 0.1],[0.0, 0.1],[0.1, 0.4],[0.05, 0.55],[0.01, 0.6],[0.01, 0.7],[0.1, 0.4],[0.05, 0.25],[0.1, 0.4],[0.08, 0.5]]]
    
    with open("open_singleshoot.json", "w") as outfile:
        for obstacle_list, guess_LagMul_scene in zip(obstacle_list_list, guess_LagMul_list):
            for init_state, guess_LagMul in zip(init_states_list, guess_LagMul_scene):
                guess_LagMul_attempt1, guess_LagMul_attempt2 = guess_LagMul
                success, elapsed_time, states, inputs, num_iter, num_notconverge = run_open(init_state, obstacle_list, guess_LagMul_attempt1, guess_LagMul_attempt2)
                cost = calculate_cost(np.array(states), np.array(inputs), obstacle_list)
                run_data = {"success": success, "cost": cost, "states": states, "inputs": inputs, "elapsed": elapsed_time, "num_iter": num_iter, "num_notconverge": num_notconverge}
                
                json.dump(run_data, outfile)
                outfile.write("\n")
                    
                if success == 1:
                    print("Successful run completed!!!\n")
                else:
                    print("Failed run with initial states {0}, obstacles {1}, and Lagrangian Multiplier guess {2}\n".format(init_state, obstacle_list, [guess_LagMul]))

    # visualize_trajectory(states, inputs, obstacle_list_list[0])

if __name__ == "__main__":
    main()