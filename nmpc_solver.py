import numpy as np
import opengen as og
import casadi.casadi as cs

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

def main(goal=[110,0]):
    global N, T
    global NX, NU, NO
    global SM_OBS
    global obstacle_list
    global lower_bounds, upper_bounds
    
    U = cs.SX.sym('U', NU * N)
    P = cs.SX.sym('P', NX + NO * 3)
    obstacle_list = P[NX:]
    
    x = cs.SX.sym('x', NX)
    u = cs.SX.sym('u', NU)
    xdot = dyn_fn(x, u, solver_opt=1)
    cost = (10.0 * u[0]**2 + 0.01 * u[1]**2 + 0.1 * x[1]**2) * T
    f = cs.Function('f', [x, u], [xdot, cost])
    
    # Init NLP
    J = 0
    g = []
    lbg = [-np.inf] * 330
    ubg = [0.0] * 330

    x_curr = P[:NX]
    for ii in range(N):
        dx, cost_togo = f(x_curr, U[NU*ii : NU*(ii+1)])
        x_next = x_curr + T * dx
        
        g = cs.vertcat(g, lower_bounds[0] - x_next[0])
        g = cs.vertcat(g, lower_bounds[1] - x_next[1])
        g = cs.vertcat(g, lower_bounds[3] - x_next[3])
        g = cs.vertcat(g, lower_bounds[7] - x_next[7])
        g = cs.vertcat(g, x_next[0] - upper_bounds[0])
        g = cs.vertcat(g, x_next[1] - upper_bounds[1])
        g = cs.vertcat(g, x_next[3] - upper_bounds[3])
        g = cs.vertcat(g, x_next[7] - upper_bounds[7])

        for jj in range(NO):
            x_obs = obstacle_list[NO*jj]
            y_obs = obstacle_list[NO*jj + 1]
            r_obs = obstacle_list[NO*jj + 2]
            g = cs.vertcat(g, (r_obs + SM_OBS)**2 - (x_next[0] - x_obs)**2 - (x_next[1] - y_obs)**2)
            J += 60.0 * (np.tanh(-1.3*((x_next[0] - x_obs)**2/(r_obs + SM_OBS)**2 + (x_next[1] - y_obs)**2/(r_obs + SM_OBS)**2)) + 1)/2
        
        J += cost_togo
        x_curr = x_next
    
    J += 100 * (((x_next[0] - goal[0])**2 + (x_next[1] - goal[1])**2) / ((goal[0] - P[0])**2 + (goal[1] - P[1])**2) + 1)
    
    set_c = og.opengen.constraints.Rectangle(lbg, ubg)
    set_y = og.opengen.constraints.Rectangle([-1e12] * len(lbg), [1e12] * len(ubg))
    
    ct_min = lower_bounds[NX:] * N
    ct_max = upper_bounds[NX:] * N
    ct_bounds = og.opengen.constraints.Rectangle(ct_min, ct_max)

    problem = og.opengen.builder.Problem(U, P, J)\
        .with_aug_lagrangian_constraints(g, set_c, set_y)\
        .with_constraints(ct_bounds)
    build_config = og.opengen.config.BuildConfiguration()\
        .with_build_directory("nmpc_solver_debug")\
        .with_build_mode("debug")\
        .with_build_python_bindings().with_rebuild(True)
    meta = og.opengen.config.OptimizerMeta()\
        .with_optimizer_name("nlmpc")
    solver_config = og.opengen.config.SolverConfiguration()\
        .with_tolerance(1e-5)\
        .with_delta_tolerance(1e-4)\
        .with_penalty_weight_update_factor(3.5)\
        .with_max_outer_iterations(20)\
        .with_lbfgs_memory(20).with_preconditioning(True)
    builder = og.opengen.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)\
        .with_verbosity_level(3)
    builder.build()

if __name__=="__main__":
    main()