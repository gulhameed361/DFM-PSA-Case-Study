
import numpy as np
import math as ma
from scipy.optimize import fsolve
from miscellaneous_X import Enthalpy_CO2
from miscellaneous_X import Enthalpy_H2O
from miscellaneous_X import Enthalpy_N2
from miscellaneous_X import Enthalpy_H2
from miscellaneous_X import Enthalpy_CH4


def simulate_hydrogenation_model(
    k7, k8, k9, k10,
    E7, E8, E10, T0, P_hyd, n,
    D, L, Time_hyd, u, epsilon,
    C_feed_hyd_H2, 
    C_CO2_init_hyd, C_H2O_init_hyd, theta_CO2_init_hyd, theta_H2O_init_hyd, T_init_hyd,
    rho, Omega, rho_s, CP_s, dp, lambda_ax, dH_CO2, dH_H2O,
    R_gas, R_, Nx, Nt
):
    dx = L / (Nx - 1)
    dt = Time_hyd / Nt

    # Allocate solution arrays: time x space
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    C_N2 = np.zeros((Nt+1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    T = np.zeros((Nt+1, Nx))
       
    uact = np.zeros((Nt+1, Nx))
    P = np.zeros((Nt+1, Nx)) + P_hyd # P in atm

    # Convert inlet pressure from atm to Pa
    P0 = P_hyd * 101325.0       # Pa

    # Inlet concentration [mmol/mL] via ideal-gas at inlet pressure
    C_init = P0/(R_gas * T0) * 1e-3
    
    # Feeding concentrations [mmol/mL]
    C0_H2 = (C_feed_hyd_H2/100.0) * C_init
    C0_N2 = C_init - C0_H2

    # Initial profile (t=0) from previous stage 
    C_CO2[0, :]     = C_CO2_init_hyd
    C_H2O[0, :]     = C_H2O_init_hyd
    C_CH4[0, :]     = 0
    C_H2[0, :]      = 0
    C_H2[0, 0]      = C0_H2
    C_N2_init_hyd   = C_init - (C_CO2_init_hyd + C_H2O_init_hyd)
    C_N2[0,:]       = C_N2_init_hyd
    C_N2[0, 0]      = C0_N2

    theta_CO2[0, :] = theta_CO2_init_hyd
    theta_H2O[0, :] = theta_H2O_init_hyd

    T[0, :] = T_init_hyd
    T[0,0] = T0

    # Residuals for implicit step
    def backward_euler_equations(y, i, t):
        C_CO2_next, C_H2O_next, C_H2_next, C_CH4_next, uact_next, theta_CO2_next, theta_H2O_next, T_next = y
        
        # ----------------- MATERIAL BALANCES  -----------------
        
        C_total_next = P[t + 1, i]/(R_ * T_next) # [mmol/mL]
        C_N2_next = C_total_next - (C_CO2_next + C_H2O_next + C_CH4_next + C_H2_next)
        
        # Equilibrium constant
        K_eq = np.exp(0.5032 * ((56000 / T_next**2) + (34633 / T_next) - (16.4 * np.log(T_next)) + (0.00557 * T_next)) + 33.165)

        # Pre-calculate rate constants
        k7_exp = k7 * np.exp(-E7 / (R_gas/1000 * T_next))
        k8_exp = k8 * np.exp(-E8 / (R_gas/1000 * T_next))
        k10_exp = k10 * np.exp(-E10 / (R_gas/1000 * T_next))
        
        P_CO2_next = (C_CO2_next / C_total_next) * P[t + 1, i]
        P_H2_next = (C_H2_next / C_total_next) * P[t + 1, i]
        P_CH4_next = (C_CH4_next / C_total_next) * P[t + 1, i]
        P_H2O_next = (C_H2O_next / C_total_next) * P[t + 1, i]
        
        # Diffusion (central / one-sided at boundaries)
        if i == Nx - 1:
            diff_CO2 = (D / epsilon) * (-C_CO2_next + C_CO2[t + 1, i - 1]) / dx**2
            diff_H2O = (D / epsilon) * (-C_H2O_next + C_H2O[t + 1, i - 1]) / dx**2
            diff_N2 = (D / epsilon) * (-C_N2_next + C_N2[t + 1, i - 1]) / dx**2
            diff_H2 = (D / epsilon) * (-C_H2_next + C_H2[t + 1, i - 1]) / dx**2
            diff_CH4 = (D / epsilon) * (-C_CH4_next + C_CH4[t + 1, i - 1]) / dx**2
            
        elif i == 1:
            diff_CO2 = (D / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diff_H2O = (D / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
            diff_N2 = (D / epsilon) * (C_N2[t, i + 1] - C_N2_next) / dx**2
            diff_H2 = (D / epsilon) * (C_H2[t, i + 1] - C_H2_next) / dx**2
            diff_CH4 = (D / epsilon) * (C_CH4[t, i + 1] - C_CH4_next) / dx**2
            
        else:
            diff_CO2 = (D / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t + 1, i - 1]) / dx**2
            diff_H2O = (D / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t + 1, i - 1]) / dx**2
            diff_N2 = (D / epsilon) * (C_N2[t, i + 1] - 2 * C_N2_next + C_N2[t + 1, i - 1]) / dx**2
            diff_H2 = (D / epsilon) * (C_H2[t, i + 1] - 2 * C_H2_next + C_H2[t + 1, i - 1]) / dx**2
            diff_CH4 = (D / epsilon) * (C_CH4[t, i + 1] - 2 * C_CH4_next + C_CH4[t + 1, i - 1]) / dx**2


        # Convection
        conv_CO2  = - (1. / epsilon) * (uact_next * C_CO2_next - uact[t + 1, i - 1] * C_CO2[t + 1, i - 1]) / dx
        conv_H2O  = - (1. / epsilon) * (uact_next * C_H2O_next - uact[t + 1, i - 1] * C_H2O[t + 1, i - 1]) / dx
        conv_N2 = - (1. / epsilon) * (uact_next * C_N2_next - uact[t + 1, i - 1] * C_N2[t + 1, i - 1]) / dx
        conv_H2  = - (1. / epsilon) * (uact_next * C_H2_next - uact[t + 1, i - 1] * C_H2[t + 1, i - 1]) / dx
        conv_CH4  = - (1. / epsilon) * (uact_next * C_CH4_next - uact[t + 1, i - 1] * C_CH4[t + 1, i - 1]) / dx
        
        
        # Reaction rate calculations at time t+1
        # Rate of CO2 desorption
        r_CO2_des_next = k7_exp * theta_CO2_next * C_H2_next
        
        # Rate of formation of CH4
        Approach_to_Equilibrium_next = P_CO2_next * (P_H2_next**4) - (P_CH4_next * P_H2O_next**2) / K_eq
        absApproach_to_Equilibrium_next = abs(Approach_to_Equilibrium_next)

        if absApproach_to_Equilibrium_next <= 1e-6:
            r_CH4_hyd_next = k8_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium_next)) * absApproach_to_Equilibrium_next, Approach_to_Equilibrium_next)
        else:
            r_CH4_hyd_next = k8_exp * ma.copysign(absApproach_to_Equilibrium_next**n, Approach_to_Equilibrium_next)

        r_H2O_ads_next = k10_exp * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) - k9 * theta_H2O_next


        # ----------------- ENERGY BALANCES -----------------
        # 1. ACCUMULATION CHANGE
        # Gas Phase:
        accumulation_change_gas =   (C_CO2_next * Enthalpy_CO2(T_next) - C_CO2[t, i] * Enthalpy_CO2(T[t, i])) + \
                                    (C_H2O_next * Enthalpy_H2O(T_next) - C_H2O[t, i] * Enthalpy_H2O(T[t, i])) + \
                                    (C_N2_next * Enthalpy_N2(T_next) - C_N2[t, i] * Enthalpy_N2(T[t, i])) + \
                                    (C_H2_next * Enthalpy_H2(T_next) - C_H2[t, i] * Enthalpy_H2(T[t, i])) + \
                                    (C_CH4_next * Enthalpy_CH4(T_next) - C_CH4[t, i] * Enthalpy_CH4(T[t, i]))

        # Solid accumulation with (1-epsilon) to represent solid volume fraction
        accumulation_change_solid = (1 - epsilon) * rho_s * CP_s * (T_next - T[t, i])
    
        # 2. CONVECTION CHANGE (Convection Rate * dt)
        convection_change = - (1. / epsilon) * ( 
            (uact_next * C_CO2_next * Enthalpy_CO2(T_next) - uact[t + 1, i - 1] * C_CO2[t + 1, i - 1] * Enthalpy_CO2(T[t + 1, i - 1])) +
            (uact_next * C_H2O_next * Enthalpy_H2O(T_next) - uact[t + 1, i - 1] * C_H2O[t + 1, i - 1] * Enthalpy_H2O(T[t + 1, i - 1])) +
            (uact_next * C_N2_next * Enthalpy_N2(T_next) - uact[t + 1, i - 1] * C_N2[t + 1, i - 1] * Enthalpy_N2(T[t + 1, i - 1])) +
            (uact_next * C_H2_next * Enthalpy_H2(T_next) - uact[t + 1, i - 1] * C_H2[t + 1, i - 1] * Enthalpy_H2(T[t + 1, i - 1])) +
            (uact_next * C_CH4_next * Enthalpy_CH4(T_next) - uact[t + 1, i - 1] * C_CH4[t + 1, i - 1] * Enthalpy_CH4(T[t + 1, i - 1]))
        ) / dx * dt
    

        # 3. DISPERSION CHANGE (Dispersion Rate * dt)
        if i == Nx - 1:
            #ADW dispersion_change = (lambda_ax / epsilon) * (T[t + 1, i - 1] - T_next) / dx**2 * dt
            dispersion_change = (lambda_ax / epsilon) * ( - T_next + T[t + 1, i - 1]) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (-C_CO2_next * Enthalpy_CO2(T_next) + C_CO2[t + 1, i - 1] * Enthalpy_CO2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (-C_H2O_next * Enthalpy_H2O(T_next) + C_H2O[t + 1, i - 1] * Enthalpy_H2O(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (-C_N2_next * Enthalpy_N2(T_next) + C_N2[t + 1, i - 1] * Enthalpy_N2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (-C_H2_next * Enthalpy_H2(T_next) + C_H2[t + 1, i - 1] * Enthalpy_H2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (-C_CH4_next * Enthalpy_CH4(T_next) + C_CH4[t + 1, i - 1] * Enthalpy_CH4(T[t + 1, i - 1])) / dx**2 * dt
            
        elif i == 1:
            dispersion_change = (lambda_ax / epsilon) * (T[t, i + 1] - T_next) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_CO2[t, i + 1] * Enthalpy_CO2(T[t, i + 1]) - C_CO2_next * Enthalpy_CO2(T_next)) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_H2O[t, i + 1] * Enthalpy_H2O(T[t, i + 1]) - C_H2O_next * Enthalpy_H2O(T_next)) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_N2[t, i + 1] * Enthalpy_N2(T[t, i + 1]) - C_N2_next * Enthalpy_N2(T_next)) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_H2[t, i + 1] * Enthalpy_H2(T[t, i + 1]) - C_H2_next * Enthalpy_H2(T_next)) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_CH4[t, i + 1] * Enthalpy_CH4(T[t, i + 1]) - C_CH4_next * Enthalpy_CH4(T_next)) / dx**2 * dt

        else:
            dispersion_change = (lambda_ax / epsilon) * (T[t, i + 1] - 2 * T_next + T[t + 1, i - 1]) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_CO2[t, i + 1] * Enthalpy_CO2(T[t, i + 1]) - 2 * C_CO2_next * Enthalpy_CO2(T_next) + C_CO2[t + 1, i - 1] *  Enthalpy_CO2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_H2O[t, i + 1] * Enthalpy_H2O(T[t, i + 1]) - 2 * C_H2O_next * Enthalpy_H2O(T_next) + C_H2O[t + 1, i - 1] *  Enthalpy_H2O(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_N2[t, i + 1] * Enthalpy_N2(T[t, i + 1]) - 2 * C_N2_next * Enthalpy_N2(T_next) + C_N2[t + 1, i - 1] *  Enthalpy_N2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_H2[t, i + 1] * Enthalpy_H2(T[t, i + 1]) - 2 * C_H2_next * Enthalpy_H2(T_next) + C_H2[t + 1, i - 1] *  Enthalpy_H2(T[t + 1, i - 1])) / dx**2 * dt
            dispersion_change = dispersion_change + (D / epsilon) * (C_CH4[t, i + 1] * Enthalpy_CH4(T[t, i + 1]) - 2 * C_CH4_next * Enthalpy_CH4(T_next) + C_CH4[t + 1, i - 1] *  Enthalpy_CH4(T[t + 1, i - 1])) / dx**2 * dt

    
        # 4. ENTHALPY CHANGE (for adsorbing/desorbing components)
        enthalpy_change = (rho * Omega) * (
            (theta_CO2_next * (Enthalpy_CO2(T_next) + dH_CO2) - theta_CO2[t, i] * (Enthalpy_CO2(T[t, i]) + dH_CO2)) +
            (theta_H2O_next * (Enthalpy_H2O(T_next) + dH_H2O) - theta_H2O[t, i] * (Enthalpy_H2O(T[t, i]) + dH_H2O)))


        # --- Residual Equations ---
        eq1 = C_CO2_next - C_CO2[t, i] - (diff_CO2 + conv_CO2) * dt - (rho * (r_CO2_des_next - r_CH4_hyd_next) * dt / epsilon)
        eq2 = C_H2_next - C_H2[t, i] - (diff_H2 + conv_H2) * dt - (rho * (-4 * r_CH4_hyd_next) * dt / epsilon)
        eq3 = C_CH4_next - C_CH4[t, i] - (diff_CH4 + conv_CH4) * dt - (rho * r_CH4_hyd_next * dt / epsilon)
        eq4 = C_H2O_next - C_H2O[t, i] - (diff_H2O + conv_H2O) * dt - (rho * (2 * r_CH4_hyd_next - r_H2O_ads_next) * dt / epsilon)
        eq5 = C_N2_next - C_N2[t, i] - (diff_N2 + conv_N2) * dt
        # Coverage factor updates
        eq6 = theta_CO2_next - theta_CO2[t, i] + (r_CO2_des_next * dt / Omega)
        eq7 = theta_H2O_next - theta_H2O[t, i] - (r_H2O_ads_next * dt / Omega)
        eq8 = (accumulation_change_gas + accumulation_change_solid / epsilon + enthalpy_change / epsilon) - (convection_change + dispersion_change)

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]


    # Time loop
    for t in range(0, Nt):

        # Boundary conditions at the inlet (i = 0)
        C_CO2[t+1, 0] = 0       # No CO2 is being fed
        C_H2[t+1, 0] = C0_H2    # H2 is being fed at the inlet
        C_CH4[t+1, 0] = 0       # No CH4 is being fed
        C_H2O[t+1, 0] = 0       # No H2O is being fed
        C_N2[t+1, 0] = C0_N2    # N2 is being fed at the inlet
        uact[t + 1, 0] = u  # cm/s
        P[t + 1, 0] = P_hyd
        T[t + 1, 0] = T0
        
        # Spatial loop (using backward Euler)
        for i in range(1, Nx):
            Gas_Density = C_CO2[t + 1, i - 1] * 44.009 + C_H2O[t + 1, i - 1] * 18.01528 + C_N2[t + 1, i - 1] * 28.0134 + C_H2[t + 1, i - 1] * 2.016 + C_CH4[t + 1, i - 1] * 16.04 #kg/m3
            Gas_Viscosity = 3.09068305480775E-07 * T[t + 1, i - 1] ** 0.713707127871733 #Pa s, results are for pure N2
            Pressure_Gradient = 150. * (1. - epsilon)**2 * Gas_Viscosity * uact[t + 1, i - 1] / (epsilon**3 * dp**2)
            Pressure_Gradient = Pressure_Gradient + 0.0001 * 1.75 * (1. - epsilon) * Gas_Density * uact[t + 1, i - 1]**2 / (epsilon**3 * dp)   #Factor of 0.0001 added to convert 1/m2 to 1/cm2 and get overall units of Pa/cm
            Pressure_Gradient = Pressure_Gradient / 101325. #Convert from Pa/cm to atm/cm
            
            if i == 1:
                P[t + 1, i] = P[t + 1, i - 1] - Pressure_Gradient * dx * 0.5
            else:
                P[t + 1, i] = P[t + 1, i - 1] - Pressure_Gradient * dx

            # Solve the implicit equations
            if i == 1: # For the first node, use the old method
                initial_guess = [C_CO2[t, i], C_H2O[t, i], C_H2[t, i], C_CH4[t, i], uact[t, i], theta_CO2[t, i], theta_H2O[t, i], T[t, i]]
            else: # For other nodes, use the solution from the previous node as the guess
                initial_guess = [C_CO2[t+1, i-1], C_H2O[t+1, i-1], C_H2[t+1, i-1], C_CH4[t+1, i-1], uact[t+1, i-1], theta_CO2[t+1, i-1], theta_H2O[t+1, i-1], T[t+1, i-1]]

            
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,t,))

            # Unpack the solution
            (C_CO2[t+1, i], C_H2O[t+1, i], C_H2[t+1, i], C_CH4[t+1, i], uact[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i], T[t + 1, i]) = solution
            C_N2[t + 1, i] = P[t + 1, i]/(R_ * T[t + 1, i])  - (C_CO2[t + 1, i] + C_H2O[t + 1, i] + C_H2[t + 1, i] + C_CH4[t + 1, i])


    return C_CO2, C_H2O, C_CH4, C_H2, theta_CO2, theta_H2O,  P, uact, T



