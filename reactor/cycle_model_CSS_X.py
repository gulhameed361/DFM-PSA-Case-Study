""" DFM CYCLE SIMULATION MODEL """


# def DFM_Model(Time_ads, Time_prg, Time_hyd, Time_prg2, F_gas, Di, T_K, C_feed_hyd_H2, P_hyd, W_DFM):
def DFM_Model(Time_hyd, F_gas, Di, max_cycles=60, export_excel=False):
    """max_cycles: CSSC loop cap (60 = full-fidelity; lower for a cheap smoke-test
    mode). export_excel: write the two detailed Excel workbooks (off by default --
    they are ~MB-scale, unconditional, and overwrite the same fixed filename on
    every call; opt in only for a diagnostic run)."""

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import perf_counter
    from adsorption_model_X import simulate_adsorption_model
    from purge_model_X import simulate_purge_model
    from hydrogenation_model_X import simulate_hydrogenation_model
    from miscellaneous_X import apply_minor_analyser_delay

    # ================================ PARAMETERS =============================== #

    # ------------------------------ Design Params ------------------------------ #

    Time_ads = 50                      # Total simulation time (s) for adsorption                         #Var
    Time_prg = 50                        # Total simulation time (s) for purge                              #Var
    Time_prg2 = 50                       # Total simulation time (s) for the 2nd purge                      #Var
    T_K = 600            # Reactor temperature set (Isothermal mode) - in K

    W_DFM = 10                          # Weight of DFM (g) 
    Ai = 3.14 * (Di**2) / 4             # Reactor (Tube) Internal Surface Area (cm2)
    u = F_gas / Ai                      # Linear velocity (cm/s) 24

    L = 5                               # Length of the Reactor (tube) (cm)

    V_react = Ai * L                    # Reactor (Tube) Volue (mL)
    rho = W_DFM/V_react                 # Adsorption bed Density (g/cm^3) 0.3-1.5  
    
    # Ce_react = (53000 + 28000*((V_react*1e-6)**0.8)) * 1 * 900/509.7                                                    #---------- Output
    # CAPEX_DFM = 2*(Ce_react + (156.2 * (W_DFM*1e-3)))                                                                   #---------- Output

    D = 0.16                            # Diffusion coefficient (cm^2/s)
    epsilon = 0.35                      # Porosity 
    P_ads = 3                           # Total pressure during adsorption (atm)                            #Var Pressures may be fixed
    P_prg = 3                           # Total pressure during purge (atm)                                 #Var Pressures may be fixed
    P_hyd = 3                           # Total pressure during hydrogenation (atm)                         #Var Pressures may be fixed

    # T_DegC  = 380                       # Reactor temperature set (Isothermal mode) - in Deg C              #Var Better to fix reactor temperature
    # T_K     = (T_DegC + 273)            # Reactor temperature set (Isothermal mode) - in K

    def calculate_omega(T_DegC):        # Fitted model to lab data for Omega vs Temperature
        Omega = (4.10e-06) * T_DegC**2 + (-4.60e-03) * T_DegC + (1.499)
        return Omega

    # Catalyst parameters
    rho_s = 3.98  # (g/cm3)
    CP_s = 0.955  # (J/g.K)
    Omega = calculate_omega(T_K - 273)
    dp = 0.1                            # cm, particle diameter (1 mm)

    # constants
    R_gas   = 8.314                     # J/(mol·K)
    R_      = 0.08206                   # (L.atm / mol.K)

    # ------------------------------ Model Params ------------------------------ #

    total_cycle_time = Time_ads + Time_prg + Time_hyd + Time_prg2

    Nx = 10                              # Number of spatial grid points

    # ---- Adaptive time-step resolution (numerical robustness) --------------
    # These stage step COUNTS used to be fixed (200/200/1000/200) regardless of
    # operating point. Since dt = T/Nt while the gas transit time across one
    # spatial cell is dx/u, a fixed count lets the Courant number
    # Cr = u*dt/dx grow without bound as Time_hyd rises or Di shrinks: the
    # scheme then advects gas many cells per step and the backward-Euler
    # fsolve degenerates, silently returning non-physical results.
    #
    # Measured at (Time_hyd=730, F_gas=6, Di=0.80), where the old Cr was 15.7:
    #     Nt_hyd=1000  (Cr=15.69) -> CH4 = -2969.37   (physically impossible)
    #     Nt_hyd=4000  (Cr= 3.92) -> CH4 =     2.20316
    #     Nt_hyd=16000 (Cr= 0.98) -> CH4 =     2.20382
    # The last two agree to 4 significant figures, i.e. grid-converged once
    # Cr <= 4. At 60 cycles that same point returned exactly 0.0, and a
    # neighbour returned 1.37e-17 with NEGATIVE H2 (-5.5e-17). Over the 192
    # cached evaluations from the first study, 105 (55%) were under-resolved
    # in hydrogenation -- including the initial point (Cr = 6.84) -- so this
    # was never confined to the corner.
    #
    # Cost is mild: 4x the steps took only 1.64x the wall time, because a
    # well-conditioned system lets fsolve converge instead of thrashing.
    # Counts are never reduced below the original defaults, so any operating
    # point that was already well resolved is unchanged.
    CR_TARGET = 4.0
    _dx_cell = L / (Nx - 1)

    def _adaptive_nt(default_nt, duration):
        if u <= 0 or duration <= 0:
            return default_nt
        need = int(np.ceil(u * duration / (_dx_cell * CR_TARGET)))
        return max(default_nt, need)

    Nt_ads = _adaptive_nt(200, Time_ads)
    Nt_prg = _adaptive_nt(200, Time_prg)
    Nt_hyd = _adaptive_nt(1000, Time_hyd)
    Nt_prg2 = _adaptive_nt(200, Time_prg2)

    C_feed_ads_CO2 = 12                 # CO2 %vol in feed gas - Adsorption stage (mmol/cm^3)   # the remaining is assumed to be CH4 as in biogas
    C_feed_hyd_H2 = 30                  # H2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 
    C_feed_ads_H2O = 0                  # H2O %vol in feed gas - Adsorption stage (mmol/cm^3)              
    C_feed_prg_CO2 = 0                  # CO2 %vol in feed gas - Purge stage (mmol/cm^3)
    C_feed_prg_H2O = 0                  # H2O %vol in feed gas - Purge stage (mmol/cm^3)

    # ----------------------------- Kinetic Params ----------------------------- #

    E6 = 34                     # Activation energy for desorption of CO2 from the adsorption sites during purge stage(J/mmol)          **(E4) in new model
    Alfa = 0.5                  # Adsorption strength correction factor
    E7 = 14                     # Activation energy for desorption of CO2 from the adsorption sites during hydrogenation stage(J/mmol)  **(E6) in new model
    E8 = 65                     # Activation energy for formation of CH4 during Hydrogenation stage (J/mmol)                            **(E5) in new model
    E10 = 10                    # Activation energy for adsorption of H2O on the adsorption sites during Hydrogenation stage (J/mmol)   **(E7) in new model

    k1 = 60                     # Optimized kinetic constant for ads. of CO2 on CO2 sites (cm3/s.g)
    k2 = 20                     # Optimized kinetic constant for ads. of H2O on CO2 sites (cm3/s.g)
    k3 = 1                      # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)
    k4 = 75.1                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **X Deleted in new model
    k5 = 19.8                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **X Deleted in new model
    k6 = 0.012                  # Optimized pre-exponential constant for purge of CO2 from CO2 sites (mmol/g.s)                         **(k4) in new model
    k7 = 38                     # Optimized kinetic constant for desorption of CO2 from CO2 sites during hydrogenation (cm3/s.g)        **(k6) in new model
    k8 = 43960                  # Optimized kinetic constant for CO2 hydrogenation (mmol/g.s.atm^5n)                                    **(k5) in new model
    k9 = 0.003                  # Optimized kinetic constant for des. of H2O from H2O sites during hydrogenation(mmol/g.s)              **(k8) in new model
    k10 = 144                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **(k7) in new model

    K_CO2 = 375.3               # Optimized adsorption constant of CO2 (cm3/mmol)                                                       **X Deleted in new model
    m = 1                       # Adjust parameter in H2O adsorption                                                                    **X Deleted in new model
    n = 0.14                    # Adjust parameter in CH4 production

    # ------------------------- Energy Balance Params ------------------------- #

    # Heat parameters for gases present in the adsorption phase
    # specific heat Capacity (J/mmol.K)
    CP_CO2 = 0.03722
    CP_H2O = 0.0347
    CP_N2 = 0.02912
    CP_CH4 = 0.0357
    CP_H2 = 0.0288

    # Thermal conductivity (W/cm.K)
    lambda_ax = 0.00045 

    # Heats of "adsorption" of the CO2 and H2O on the surface of the material
    # Enthalpy (J/mmol)
    dH_CO2 = -120
    dH_H2O = -90

    # Heat of formation at reference temperature of 0 K (J/mmol)
    dHf_CO2 = -393.5 - 298.15 * CP_CO2
    dHf_H2O = -241.818 - 298.15 * CP_H2O
    dHf_N2 = 0. - 298.15 * CP_N2
    dHf_CH4 = -74.87 - 298.15 * CP_CH4
    dHf_H2 = 0.0 - 298.15 * CP_H2

    # Create functions to calculate the enthalpy of the CO2, H2O and N2 in the gaseous state with respect to temperature
    def Enthalpy_CO2(Temperature_K):
        return dHf_CO2 + CP_CO2 * Temperature_K;

    def Enthalpy_H2O(Temperature_K):
        return dHf_H2O + CP_H2O * Temperature_K;

    def Enthalpy_N2(Temperature_K):
        return dHf_N2 + CP_N2 * Temperature_K;

    def Enthalpy_H2(Temperature_K):
        return dHf_H2 + CP_H2 * Temperature_K;

    def Enthalpy_CH4(Temperature_K):
        return dHf_CH4 + CP_CH4 * Temperature_K;

    # ---------------------------- Analyser Params ---------------------------- #

    tau_increase = 20            # Time constant for increasing step (Analyser delay) 19.2                  #Var
    tau_decrease = 20            # Time constant for decreasing step (Analyser delay) 22                    #Var

    # ================================ SIMULATION =============================== #

    def run_cyclic_process(tol=1e-4, max_cycles=60):
        """Run the cyclic process until CSSC occurs. Return data of the steady cycle."""
        C_CO2, C_H2O, theta_CO2, theta_H2O, T = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
        C_CO2_init_ads, C_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)             #initial concentration of CO2 and H2O inside the column (before running the cycle)
        theta_CO2_init_ads, theta_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)     #initial coverage area for CO2 and H2O inside the column (before running the cycle)
        T_init_ads = np.zeros(Nx) + T_K             #initial temperature inside the column (before running the cycle)
        T = T_init_ads
        all_cycles = []

        for cycle in range(max_cycles):
            # print(f"Cycle {cycle+1}")
            # print("Adsorption")

            # Adsorption
            C_CO2_ads, C_H2O_ads, theta_CO2_ads, theta_H2O_ads, P_profile_ads, uact_profile_ads, T_profile_ads = simulate_adsorption_model(
                k1, k2, k3, k4, k5, K_CO2, m,
                T_K, P_ads, D, L, Time_ads, u, epsilon,
                C_feed_ads_CO2, C_feed_ads_H2O,
                C_CO2_init_ads, C_H2O_init_ads,
                theta_CO2_init_ads, theta_H2O_init_ads, T_init_ads,
                rho, Omega, rho_s, CP_s, dp, lambda_ax, dH_CO2, dH_H2O,
                R_gas, R_, Nx, Nt_ads
            )

            # print("Purge")
            # Purge
            C_CO2_prg, C_H2O_prg, C_CH4_prg, C_H2_prg, theta_CO2_prg, theta_H2O_prg, P_profile_prg, uact_profile_prg, T_profile_prg= simulate_purge_model(
                k2, k3, k4, k5, k6, K_CO2, m,
                E6, T_K, P_prg, Alfa,
                D, L, Time_prg, u, epsilon,
                C_feed_prg_CO2, C_feed_prg_H2O, 
                C_CO2_ads[-1, :], C_H2O_ads[-1, :], 0, 0,
                theta_CO2_ads[-1, :], theta_H2O_ads[-1, :], T_profile_ads[-1, :],
                rho, Omega, rho_s, CP_s, dp, lambda_ax, dH_CO2, dH_H2O,
                R_gas, R_, Nx, Nt_prg
            )

            # print("Hydrogenation")
            # Hydrogenation
            C_CO2_hyd, C_H2O_hyd, C_CH4_hyd, C_H2_hyd, theta_CO2_hyd, theta_H2O_hyd, P_profile_hyd, uact_profile_hyd, T_profile_hyd= simulate_hydrogenation_model(
                k7, k8, k9, k10,
                E7, E8, E10, T_K, P_hyd, n,
                D, L, Time_hyd, u, epsilon,
                C_feed_hyd_H2, 
                C_CO2_prg[-1, :], C_H2O_prg[-1, :], theta_CO2_prg[-1, :], theta_H2O_prg[-1, :], T_profile_prg[-1, :],
                rho, Omega, rho_s, CP_s, dp, lambda_ax, dH_CO2, dH_H2O,
                R_gas, R_, Nx, Nt_hyd
            )

            # print("Final Purge")
            # Final Purge
            C_CO2_prg2, C_H2O_prg2, C_CH4_prg2, C_H2_prg2, theta_CO2_prg2, theta_H2O_prg2, P_profile_prg2, uact_profile_prg2, T_profile_prg2= simulate_purge_model(
                k2, k3, k4, k5, k6, K_CO2, m,
                E6, T_K, P_prg, Alfa,
                D, L, Time_prg2, u, epsilon,
                C_feed_prg_CO2, C_feed_prg_H2O, 
                C_CO2_hyd[-1, :], C_H2O_hyd[-1, :], C_CH4_hyd[-1, :], C_H2_hyd[-1, :],
                theta_CO2_hyd[-1, :], theta_H2O_hyd[-1, :], T_profile_hyd[-1, :],
                rho, Omega, rho_s, CP_s, dp, lambda_ax, dH_CO2, dH_H2O,
                R_gas, R_, Nx, Nt_prg2
            )

            # Store data for convergence plot
            all_cycles.append({
                "C_CO2_ads": C_CO2_ads,
                "C_H2O_ads": C_H2O_ads,
                "theta_CO2_ads": theta_CO2_ads,
                "theta_H2O_ads": theta_H2O_ads,
                "C_CO2_prg": C_CO2_prg,
                "C_H2O_prg": C_H2O_prg,
                "C_CH4_prg": C_CH4_prg,      
                "C_H2_prg": C_H2_prg,        
                "theta_CO2_prg": theta_CO2_prg,
                "theta_H2O_prg": theta_H2O_prg,
                "C_CO2_hyd": C_CO2_hyd,
                "C_H2O_hyd": C_H2O_hyd,
                "C_CH4_hyd": C_CH4_hyd,
                "C_H2_hyd": C_H2_hyd,
                "theta_CO2_hyd": theta_CO2_hyd,
                "theta_H2O_hyd": theta_H2O_hyd,
                "C_CO2_prg2": C_CO2_prg2,
                "C_H2O_prg2": C_H2O_prg2,
                "C_CH4_prg2": C_CH4_prg2,
                "C_H2_prg2": C_H2_prg2,
                "theta_CO2_prg2": theta_CO2_prg2,
                "theta_H2O_prg2": theta_H2O_prg2,
                "T_profile_ads": T_profile_ads,
                "T_profile_prg": T_profile_prg,
                "T_profile_hyd": T_profile_hyd,
                "T_profile_prg2": T_profile_prg2,
            })

            # Convergence check
            if (
                np.all(np.abs(C_CO2_prg2[-1, :] - C_CO2) < tol) and
                np.all(np.abs(C_H2O_prg2[-1, :] - C_H2O) < tol) and
                np.all(np.abs(theta_CO2_prg2[-1, :] - theta_CO2) < 0.05) and
                np.all(np.abs(theta_H2O_prg2[-1, :] - theta_H2O) < 0.05) and
                np.all(np.abs(T_profile_prg2[-1, :] - T_init_ads) < 0.1)
            ):
                # print(f"Converged to CSSC in {cycle+1} cycles.")
                return {
                    "steady_cycle": {
                        "C_CO2_ads": C_CO2_ads,
                        "C_H2O_ads": C_H2O_ads,
                        "theta_CO2_ads": theta_CO2_ads,
                        "theta_H2O_ads": theta_H2O_ads,
                        "C_CO2_prg": C_CO2_prg,
                        "C_H2O_prg": C_H2O_prg,
                        "C_CH4_prg": C_CH4_prg,      
                        "C_H2_prg": C_H2_prg,        
                        "theta_CO2_prg": theta_CO2_prg,
                        "theta_H2O_prg": theta_H2O_prg,
                        "C_CO2_hyd": C_CO2_hyd,
                        "C_H2O_hyd": C_H2O_hyd,
                        "C_CH4_hyd": C_CH4_hyd,
                        "C_H2_hyd": C_H2_hyd,
                        "theta_CO2_hyd": theta_CO2_hyd,
                        "theta_H2O_hyd": theta_H2O_hyd,
                        "C_CO2_prg2": C_CO2_prg2,
                        "C_H2O_prg2": C_H2O_prg2,
                        "C_CH4_prg2": C_CH4_prg2,
                        "C_H2_prg2": C_H2_prg2,
                        "theta_CO2_prg2": theta_CO2_prg2,
                        "theta_H2O_prg2": theta_H2O_prg2,
                        "T_profile_ads": T_profile_ads,
                        "T_profile_prg": T_profile_prg,
                        "T_profile_hyd": T_profile_hyd,
                        "T_profile_prg2": T_profile_prg2,
                    },
                    "all_cycles": all_cycles,
                }

            # Update for next cycle
            C_CO2, C_H2O = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
            theta_CO2, theta_H2O = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
            T = T_profile_prg[-1, :]
            C_CO2_init_ads, C_H2O_init_ads = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
            theta_CO2_init_ads, theta_H2O_init_ads = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
            T_init_ads = T_profile_prg2[-1, :]

        print("Failed to converge to CSSC within the maximum number of cycles. Returning data from the last cycle.")
        return {
            "steady_cycle": {
                "C_CO2_ads": C_CO2_ads,
                "C_H2O_ads": C_H2O_ads,
                "theta_CO2_ads": theta_CO2_ads,
                "theta_H2O_ads": theta_H2O_ads,
                "C_CO2_prg": C_CO2_prg,
                "C_H2O_prg": C_H2O_prg,
                "C_CH4_prg": C_CH4_prg,
                "C_H2_prg": C_H2_prg,
                "theta_CO2_prg": theta_CO2_prg,
                "theta_H2O_prg": theta_H2O_prg,
                "C_CO2_hyd": C_CO2_hyd,
                "C_H2O_hyd": C_H2O_hyd,
                "C_CH4_hyd": C_CH4_hyd,
                "C_H2_hyd": C_H2_hyd,
                "theta_CO2_hyd": theta_CO2_hyd,
                "theta_H2O_hyd": theta_H2O_hyd,
                "C_CO2_prg2": C_CO2_prg2,
                "C_H2O_prg2": C_H2O_prg2,
                "C_CH4_prg2": C_CH4_prg2,
                "C_H2_prg2": C_H2_prg2,
                "theta_CO2_prg2": theta_CO2_prg2,
                "theta_H2O_prg2": theta_H2O_prg2,
                "T_profile_ads": T_profile_ads,
                "T_profile_prg": T_profile_prg,
                "T_profile_hyd": T_profile_hyd,
                "T_profile_prg2": T_profile_prg2,
            },
            "all_cycles": all_cycles,
        }

    # =========================== PERFORMANCE METRICS ========================== #

    def calculate_recovery(C_feed_ads_CO2, C_CH4_hyd, F_gas, Time_ads, Time_hyd):
        """Calculate recovery using steady-state data."""
        dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
        total_CH4_out = np.trapezoid(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
        total_CO2_in = F_gas * (P_ads * C_feed_ads_CO2/(R_ * T_K * 100)) * Time_ads     # Total CO2 fed during adsorption
        Recovery = total_CH4_out / total_CO2_in
        return Recovery, total_CH4_out, total_CO2_in


    def calculate_productivity(C_CH4_hyd, F_gas, W_DFM, Time_cycle, Time_hyd):
        """Calculate productivity using steady-state data."""
        dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
        total_CH4_out = np.trapezoid(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
        Productivity = total_CH4_out / (W_DFM * Time_cycle)                             # Productivity
        return Productivity, total_CH4_out


    def calculate_purity(C_CH4_hyd, C_H2_hyd, C_CO2_hyd, F_gas, Time_hyd):
        """Calculate purity using steady-state data."""
        dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
        total_CH4_out = np.trapezoid(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
        total_hydrogenation_out = np.trapezoid(F_gas * (C_CH4_hyd + C_H2_hyd + C_CO2_hyd), dx=dt_hyd)  # Total gas output (except water & N2)
        total_H2_out = np.trapezoid(F_gas * C_H2_hyd, dx=dt_hyd)
        total_CO2_out = np.trapezoid(F_gas * C_CO2_hyd, dx=dt_hyd)
        Purity = total_CH4_out / total_hydrogenation_out
        return Purity, total_CH4_out, total_H2_out, total_CO2_out, total_hydrogenation_out


    # ============================= PLOTTING RESULTS ============================ #

    def plot_convergence(all_cycles, Time_ads, Time_prg, Time_hyd, Time_prg2, Nt_ads, Nt_prg, Nt_hyd, Nt_prg2, file_name="convergence_data.xlsx"):
        """Plot and export the convergence of concentrations and theta values over cycles."""
        time_ads = np.linspace(0, Time_ads, Nt_ads + 1)
        time_prg = np.linspace(Time_ads, Time_ads + Time_prg, Nt_prg + 1)
        time_hyd = np.linspace(Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd, Nt_hyd + 1)
        time_prg2 = np.linspace(Time_ads + Time_prg + Time_hyd, Time_ads + Time_prg + Time_hyd + Time_prg2, Nt_prg2 + 1)

        # Full cycle time
        time_cycle = np.concatenate((time_ads, time_prg, time_hyd, time_prg2))
        convergence_data = []
        # Plot concentrations
        components = ["CO2", "H2O", "CH4", "H2"]
        for comp in components:
            plt.figure(figsize=(12, 8))
            for cycle_idx, cycle_data in enumerate(all_cycles):
                if comp == "CO2":
                    C_ads_out = cycle_data["C_CO2_ads"][:, -1]
                    C_prg_out = cycle_data["C_CO2_prg"][:, -1]
                    C_hyd_out = cycle_data["C_CO2_hyd"][:, -1]
                    C_prg2_out = cycle_data["C_CO2_prg2"][:, -1]
                elif comp == "H2O":
                    C_ads_out = cycle_data["C_H2O_ads"][:, -1]
                    C_prg_out = cycle_data["C_H2O_prg"][:, -1]
                    C_hyd_out = cycle_data["C_H2O_hyd"][:, -1]
                    C_prg2_out = cycle_data["C_H2O_prg2"][:, -1]
                elif comp == "CH4":
                    C_ads_out = np.zeros_like(time_ads)
                    C_prg_out = np.zeros_like(time_prg)
                    C_hyd_out = cycle_data["C_CH4_hyd"][:, -1]
                    C_prg2_out = cycle_data["C_CH4_prg2"][:, -1]
                elif comp == "H2":
                    C_ads_out = np.zeros_like(time_ads)
                    C_prg_out = np.zeros_like(time_prg)
                    C_hyd_out = cycle_data["C_H2_hyd"][:, -1]
                    C_prg2_out = cycle_data["C_H2_prg2"][:, -1]

                C_cycle = np.concatenate((C_ads_out, C_prg_out, C_hyd_out, C_prg2_out))
                plt.plot(time_cycle, C_cycle, label=f"Cycle {cycle_idx+1}")
                convergence_data.append({
                "Cycle": cycle_idx + 1,
                "Component": comp,
                "Time (s)": list(time_cycle),
                "Concentration": list(C_cycle)
                })
           
            # Dashed lines for stages
            for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
                plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)

            plt.title(f"{comp} Convergence Across Cycles", fontsize=16)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel(f"{comp} Concentration (mmol/cm³)", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
        
        # Plot theta values
        for theta, label in zip(["theta_CO2", "theta_H2O"], ["Theta CO2", "Theta H2O"]):
            plt.figure(figsize=(12, 8))
            for cycle_idx, cycle_data in enumerate(all_cycles):
                theta_ads = cycle_data[f"{theta}_ads"][:, -1]
                theta_prg = cycle_data[f"{theta}_prg"][:, -1]
                theta_hyd = cycle_data[f"{theta}_hyd"][:, -1]
                theta_prg2 = cycle_data[f"{theta}_prg2"][:, -1]

                theta_cycle = np.concatenate((theta_ads, theta_prg, theta_hyd, theta_prg2))
                plt.plot(time_cycle, theta_cycle, label=f"Cycle {cycle_idx+1}")

            # Dashed lines for stages
            for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
                plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)

            plt.title(f"{label} Convergence Across Cycles", fontsize=16)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel("Theta Values", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
        
        # Export convergence data to Excel
        df_convergence = pd.json_normalize(convergence_data, sep="_")
        df_convergence.to_excel(file_name, sheet_name="Convergence Data", index=False)
        # print(f"Convergence data exported to {file_name}")


    def plot_final_cycle(steady_cycle, Time_ads, Time_prg, Time_hyd, Time_prg2, Nt_ads, Nt_prg, Nt_hyd, Nt_prg2, file_name="steady_cycle_data.xlsx"):
        """Plot the final cycle with separate plots for concentrations and theta values."""
        time_ads = np.linspace(0, Time_ads, Nt_ads + 1)
        time_prg = np.linspace(Time_ads, Time_ads + Time_prg, Nt_prg + 1)
        time_hyd = np.linspace(Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd, Nt_hyd + 1)
        time_prg2 = np.linspace(Time_ads + Time_prg + Time_hyd, Time_ads + Time_prg + Time_hyd + Time_prg2, Nt_prg2 + 1)

        time_cycle = np.concatenate((time_ads, time_prg, time_hyd, time_prg2))

        # Apply analyser delay effect
        dt_ads = Time_ads / Nt_ads
        dt_prg = Time_prg / Nt_prg
        dt_hyd = Time_hyd / Nt_hyd
        dt_prg2 = Time_prg2 / Nt_prg2
        
        C_CO2_ads_measured = apply_minor_analyser_delay(steady_cycle["C_CO2_ads"][:, -1], dt_ads, tau_increase, tau_decrease)
        C_CO2_prg_measured = apply_minor_analyser_delay(steady_cycle["C_CO2_prg"][:, -1], dt_prg, tau_increase, tau_decrease)
        C_CO2_hyd_measured = apply_minor_analyser_delay(steady_cycle["C_CO2_hyd"][:, -1], dt_hyd, tau_increase, tau_decrease)
        C_CO2_prg2_measured = apply_minor_analyser_delay(steady_cycle["C_CO2_prg2"][:, -1], dt_prg2, tau_increase, tau_decrease)
        
        C_H2O_ads_measured = apply_minor_analyser_delay(steady_cycle["C_H2O_ads"][:, -1], dt_ads, tau_increase, tau_decrease)
        C_H2O_prg_measured = apply_minor_analyser_delay(steady_cycle["C_H2O_prg"][:, -1], dt_prg, tau_increase, tau_decrease)
        C_H2O_hyd_measured = apply_minor_analyser_delay(steady_cycle["C_H2O_hyd"][:, -1], dt_hyd, tau_increase, tau_decrease)
        C_H2O_prg2_measured = apply_minor_analyser_delay(steady_cycle["C_H2O_prg2"][:, -1], dt_prg2, tau_increase, tau_decrease)

        C_CH4_hyd_measured = apply_minor_analyser_delay(steady_cycle["C_CH4_hyd"][:, -1], dt_hyd, tau_increase, tau_decrease)
        C_CH4_prg2_measured = apply_minor_analyser_delay(steady_cycle["C_CH4_prg2"][:, -1], dt_prg2, tau_increase, tau_decrease)
        C_H2_hyd_measured = apply_minor_analyser_delay(steady_cycle["C_H2_hyd"][:, -1], dt_hyd, tau_increase, tau_decrease)
        C_H2_prg2_measured = apply_minor_analyser_delay(steady_cycle["C_H2_prg2"][:, -1], dt_prg2, tau_increase, tau_decrease)


        # Concentrations
        C_CO2_cycle = np.concatenate((C_CO2_ads_measured,
                                       C_CO2_prg_measured,
                                       C_CO2_hyd_measured,
                                       C_CO2_prg2_measured))
        C_H2O_cycle = np.concatenate((C_H2O_ads_measured,
                                       C_H2O_prg_measured,
                                       C_H2O_hyd_measured,
                                       C_H2O_prg2_measured))
        C_CH4_cycle = np.concatenate((np.zeros_like(time_ads),
                                       np.zeros_like(time_prg),
                                       C_CH4_hyd_measured,
                                       C_CH4_prg2_measured))
        C_H2_cycle = np.concatenate((np.zeros_like(time_ads),
                                       np.zeros_like(time_prg),
                                       C_H2_hyd_measured,
                                       C_H2_prg2_measured))
       
        # Plot concentrations
        plt.figure(figsize=(12, 8))
        plt.plot(time_cycle, C_CO2_cycle, label="CO2 Concentration", color="red")
        plt.plot(time_cycle, C_H2O_cycle, label="H2O Concentration", color="blue")
        plt.plot(time_cycle, C_CH4_cycle, label="CH4 Concentration", color="green")
        plt.plot(time_cycle, C_H2_cycle, label="H2 Concentration", color="orange")
        for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
            plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)
        plt.title("Final Cycle: Concentration Profiles", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Concentration (mmol/cm³)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Theta values
        theta_CO2_cycle = np.concatenate((steady_cycle["theta_CO2_ads"][:, -1],
                                           steady_cycle["theta_CO2_prg"][:, -1],
                                           steady_cycle["theta_CO2_hyd"][:, -1],
                                           steady_cycle["theta_CO2_prg2"][:, -1]))
        theta_H2O_cycle = np.concatenate((steady_cycle["theta_H2O_ads"][:, -1],
                                           steady_cycle["theta_H2O_prg"][:, -1],
                                           steady_cycle["theta_H2O_hyd"][:, -1],
                                           steady_cycle["theta_H2O_prg2"][:, -1]))

        plt.figure(figsize=(12, 8))
        plt.plot(time_cycle, theta_CO2_cycle, label="Theta CO2", linestyle="--", color="purple")
        plt.plot(time_cycle, theta_H2O_cycle, label="Theta H2O", linestyle="--", color="orange")
        for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
            plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)
        plt.title("Final Cycle: Theta Profiles", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Theta Values", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        final_cycle_data = []
        final_cycle_data.append({
            "Time (s)": list(time_cycle),
            "CO2 Concentration": list(C_CO2_cycle),
            "H2O Concentration": list(C_H2O_cycle),
            "CH4 Concentration": list(C_CH4_cycle),
            "H2 Concentration": list(C_H2_cycle),
            "Theta CO2": list(theta_CO2_cycle),
            "Theta H2O": list(theta_H2O_cycle)    
            })
        # Export steady cycle data to Excel
        df_final_cycle = pd.json_normalize(final_cycle_data, sep="_")
        df_final_cycle.to_excel(file_name, sheet_name="Steady Cycle Data", index=False)

        # print(f"Steady cycle data exported to {file_name}")


    def plot_temperature_profiles(all_cycles, steady_cycle, Time_ads, Time_prg, Time_hyd, Time_prg2, Nt_ads, Nt_prg, Nt_hyd, Nt_prg2):
        """
        Plots the convergence and final steady-state profile for outlet temperature.
        """
        # --- Define Time Arrays ---
        time_ads = np.linspace(0, Time_ads, Nt_ads + 1)
        time_prg = np.linspace(Time_ads, Time_ads + Time_prg, Nt_prg + 1)
        time_hyd = np.linspace(Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd, Nt_hyd + 1)
        time_prg2 = np.linspace(Time_ads + Time_prg + Time_hyd, total_cycle_time, Nt_prg2 + 1)
        time_cycle = np.concatenate((time_ads, time_prg[1:], time_hyd[1:], time_prg2[1:]))

        # --- 1. Plot Temperature Convergence ---
        plt.figure(figsize=(12, 8))
        for cycle_idx, cycle_data in enumerate(all_cycles):
            T_ads = cycle_data["T_profile_ads"][:, -1]
            T_prg = cycle_data["T_profile_prg"][:, -1]
            T_hyd = cycle_data["T_profile_hyd"][:, -1]
            T_prg2 = cycle_data["T_profile_prg2"][:, -1]

            T_cycle = np.concatenate((T_ads, T_prg[1:], T_hyd[1:], T_prg2[1:]))
            plt.plot(time_cycle, T_cycle, label=f"Cycle {cycle_idx+1}")

        for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
            plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)

        plt.title("Outlet Temperature Convergence Across Cycles", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Temperature (K)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # --- 2. Plot Final Cycle Temperature ---
        T_ads_out = steady_cycle["T_profile_ads"][:, -1]
        T_prg_out = steady_cycle["T_profile_prg"][:, -1]
        T_hyd_out = steady_cycle["T_profile_hyd"][:, -1]
        T_prg2_out = steady_cycle["T_profile_prg2"][:, -1]
        T_cycle_final = np.concatenate((T_ads_out, T_prg_out[1:], T_hyd_out[1:], T_prg2_out[1:]))

        plt.figure(figsize=(12, 8))
        plt.plot(time_cycle, T_cycle_final, label="Outlet Temperature", color="magenta")
        for stage_end in [Time_ads, Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd]:
            plt.axvline(stage_end, color="black", linestyle="--", linewidth=1.5)
        plt.title("Final Cycle: Outlet Temperature Profile", fontsize=16)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Temperature (K)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ============================= EXPORTING RESULTS ============================ #

    def export_detailed_data(all_cycles, steady_cycle, Nx, Time_ads, Time_prg, Time_hyd, Time_prg2, Nt_ads, Nt_prg, Nt_hyd, Nt_prg2):
        """
        Exports detailed data for specific locations within the reactor to Excel files
        for both convergence and the final steady cycle.
        """
        # print("\nExporting detailed data to Excel...")

        # --- Define Locations and Time Arrays ---
        locations = {
            "Middle_Reactor": int((Nx - 1) / 2),
            "3_4_Reactor": int(3 * (Nx - 1) / 4),
            "End_Reactor": Nx - 1
        }

        time_ads = np.linspace(0, Time_ads, Nt_ads + 1)
        time_prg = np.linspace(Time_ads, Time_ads + Time_prg, Nt_prg + 1)
        time_hyd = np.linspace(Time_ads + Time_prg, Time_ads + Time_prg + Time_hyd, Nt_hyd + 1)
        time_prg2 = np.linspace(Time_ads + Time_prg + Time_hyd, Time_ads + Time_prg + Time_hyd + Time_prg2, Nt_prg2 + 1)
        
        # Use [1:] for subsequent stages to avoid duplicating the endpoint of the previous stage
        time_cycle = np.concatenate((time_ads, time_prg[1:], time_hyd[1:], time_prg2[1:]))

        # --- 1. Export Convergence Data for All Cycles ---
        with pd.ExcelWriter("convergence_data_detailed.xlsx", engine='xlsxwriter') as writer:
            for cycle_idx, cycle_data in enumerate(all_cycles):
                for loc_name, loc_idx in locations.items():
                    
                    # --- Gather data for each component at the specified location ---
                    C_CO2_cycle = np.concatenate((
                        cycle_data["C_CO2_ads"][:, loc_idx], cycle_data["C_CO2_prg"][:, loc_idx][1:],
                        cycle_data["C_CO2_hyd"][:, loc_idx][1:], cycle_data["C_CO2_prg2"][:, loc_idx][1:]
                    ))
                    C_H2O_cycle = np.concatenate((
                        cycle_data["C_H2O_ads"][:, loc_idx], cycle_data["C_H2O_prg"][:, loc_idx][1:],
                        cycle_data["C_H2O_hyd"][:, loc_idx][1:], cycle_data["C_H2O_prg2"][:, loc_idx][1:]
                    ))
                    C_CH4_cycle = np.concatenate((
                        np.zeros(Nt_ads + 1), cycle_data["C_CH4_prg"][:, loc_idx][1:],
                        cycle_data["C_CH4_hyd"][:, loc_idx][1:], cycle_data["C_CH4_prg2"][:, loc_idx][1:]
                    ))
                    C_H2_cycle = np.concatenate((
                        np.zeros(Nt_ads + 1), cycle_data["C_H2_prg"][:, loc_idx][1:],
                        cycle_data["C_H2_hyd"][:, loc_idx][1:], cycle_data["C_H2_prg2"][:, loc_idx][1:]
                    ))
                    theta_CO2_cycle = np.concatenate((
                        cycle_data["theta_CO2_ads"][:, loc_idx], cycle_data["theta_CO2_prg"][:, loc_idx][1:],
                        cycle_data["theta_CO2_hyd"][:, loc_idx][1:], cycle_data["theta_CO2_prg2"][:, loc_idx][1:]
                    ))
                    theta_H2O_cycle = np.concatenate((
                        cycle_data["theta_H2O_ads"][:, loc_idx], cycle_data["theta_H2O_prg"][:, loc_idx][1:],
                        cycle_data["theta_H2O_hyd"][:, loc_idx][1:], cycle_data["theta_H2O_prg2"][:, loc_idx][1:]
                    ))
                    T_cycle = np.concatenate((
                        cycle_data["T_profile_ads"][:, loc_idx], cycle_data["T_profile_prg"][:, loc_idx][1:],
                        cycle_data["T_profile_hyd"][:, loc_idx][1:], cycle_data["T_profile_prg2"][:, loc_idx][1:]
                    ))

                    # --- Create DataFrame and write to sheet ---
                    df_data = {
                        "Time_s": time_cycle, "Temp_K": T_cycle, "C_CO2": C_CO2_cycle, "C_H2O": C_H2O_cycle,
                        "C_CH4": C_CH4_cycle, "C_H2": C_H2_cycle, "theta_CO2": theta_CO2_cycle, "theta_H2O": theta_H2O_cycle
                    }
                    df = pd.DataFrame(df_data)
                    sheet_name = f"C{cycle_idx+1}_{loc_name}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        # --- 2. Export Data for the Final Steady Cycle (with Analyser Delay---
        with pd.ExcelWriter("steady_cycle_data_detailed.xlsx", engine='xlsxwriter') as writer:
            for loc_name, loc_idx in locations.items():
                # --- Gather the raw model output first ---
                C_CO2_raw = np.concatenate((
                    steady_cycle["C_CO2_ads"][:, loc_idx], steady_cycle["C_CO2_prg"][:, loc_idx][1:],
                    steady_cycle["C_CO2_hyd"][:, loc_idx][1:], steady_cycle["C_CO2_prg2"][:, loc_idx][1:]
                ))
                C_H2O_raw = np.concatenate((
                    steady_cycle["C_H2O_ads"][:, loc_idx], steady_cycle["C_H2O_prg"][:, loc_idx][1:],
                    steady_cycle["C_H2O_hyd"][:, loc_idx][1:], steady_cycle["C_H2O_prg2"][:, loc_idx][1:]
                ))
                C_CH4_raw = np.concatenate((
                    np.zeros(Nt_ads + 1), steady_cycle["C_CH4_prg"][:, loc_idx][1:],
                    steady_cycle["C_CH4_hyd"][:, loc_idx][1:], steady_cycle["C_CH4_prg2"][:, loc_idx][1:]
                ))
                C_H2_raw = np.concatenate((
                    np.zeros(Nt_ads + 1), steady_cycle["C_H2_prg"][:, loc_idx][1:],
                    steady_cycle["C_H2_hyd"][:, loc_idx][1:], steady_cycle["C_H2_prg2"][:, loc_idx][1:]
                ))
                theta_CO2_cycle = np.concatenate((
                    steady_cycle["theta_CO2_ads"][:, loc_idx], steady_cycle["theta_CO2_prg"][:, loc_idx][1:],
                    steady_cycle["theta_CO2_hyd"][:, loc_idx][1:], steady_cycle["theta_CO2_prg2"][:, loc_idx][1:]
                ))
                theta_H2O_cycle = np.concatenate((
                    steady_cycle["theta_H2O_ads"][:, loc_idx], steady_cycle["theta_H2O_prg"][:, loc_idx][1:],
                    steady_cycle["theta_H2O_hyd"][:, loc_idx][1:], steady_cycle["theta_H2O_prg2"][:, loc_idx][1:]
                ))
                T_cycle = np.concatenate((
                    steady_cycle["T_profile_ads"][:, loc_idx], steady_cycle["T_profile_prg"][:, loc_idx][1:],
                    steady_cycle["T_profile_hyd"][:, loc_idx][1:], steady_cycle["T_profile_prg2"][:, loc_idx][1:]
                ))
                
                # --- Apply the analyzer delay to concentration profiles ---
                total_time = Time_ads + Time_prg + Time_hyd + Time_prg2
                total_steps = len(time_cycle) - 1
                avg_dt = total_time / total_steps
                
                C_CO2_measured = apply_minor_analyser_delay(C_CO2_raw, avg_dt, tau_increase=tau_increase, tau_decrease=tau_decrease)
                C_H2O_measured = apply_minor_analyser_delay(C_H2O_raw, avg_dt, tau_increase=tau_increase, tau_decrease=tau_decrease)
                C_H2_measured = apply_minor_analyser_delay(C_H2_raw, avg_dt, tau_increase=tau_increase, tau_decrease=tau_decrease)
                C_CH4_measured = apply_minor_analyser_delay(C_CH4_raw, avg_dt, tau_increase=tau_increase, tau_decrease=tau_decrease)

                # --- Create DataFrame with the DELAYED concentrations ---
                df_data = {
                    "Time_s": time_cycle, 
                    "Temp_K": T_cycle, 
                    "C_CO2_measured": C_CO2_measured, 
                    "C_H2O_measured": C_H2O_measured,
                    "C_CH4_measured": C_CH4_measured, 
                    "C_H2_measured": C_H2_measured, 
                    "theta_CO2": theta_CO2_cycle, 
                    "theta_H2O": theta_H2O_cycle
                }
                df = pd.DataFrame(df_data)
                df.to_excel(writer, sheet_name=loc_name, index=False)
                         
        # print("Detailed data export complete.")


    start_time = perf_counter()
        
    # Run the cyclic process to reach CSSC
    results = run_cyclic_process(tol=1e-5, max_cycles=max_cycles)
    if results:
        all_cycles = results["all_cycles"]
        steady_cycle = results["steady_cycle"]

        if export_excel:
            export_detailed_data(
                    all_cycles,
                    steady_cycle,
                    Nx,
                    Time_ads, Time_prg, Time_hyd, Time_prg2,
                    Nt_ads, Nt_prg, Nt_hyd, Nt_prg2
            )

        # Extract steady-state data for performance metrics (uses outlet data: [:, -1])
        C_CO2_ads = steady_cycle["C_CO2_ads"][:, -1]
        C_CH4_hyd = steady_cycle["C_CH4_hyd"][:, -1]
        C_H2_hyd = steady_cycle["C_H2_hyd"][:, -1]
        C_CO2_hyd = steady_cycle["C_CO2_hyd"][:, -1]

        # Calculate metrics
        recovery, total_CH4_out, total_CO2_in = calculate_recovery(
                C_feed_ads_CO2, C_CH4_hyd, F_gas, Time_ads, Time_hyd,
        )
        productivity, total_CH4_out = calculate_productivity(
                C_CH4_hyd, F_gas, W_DFM, total_cycle_time, Time_hyd,
        )
        purity, total_CH4_out, total_H2_out, total_CO2_out, total_hydrogenation_out = calculate_purity(
                C_CH4_hyd, C_H2_hyd, C_CO2_hyd, F_gas, Time_hyd,
        )


        # Measure execution time
    duration = perf_counter() - start_time
        
    # print(f"Total cycle time: {total_cycle_time:.2f} seconds")
    # print(f"Recovery: {recovery*100:.10f}%")
    # print(f"Productivity: {productivity*1000:.10f} mmol/kg_DFM/cycle")
    # print(f"Purity: {purity*100:.10f}%")
    # print(f"Total CO2 adsorbed: {total_CO2_in:.6f} mmol")
    # print(f"Total CH4 at DFM outlet: {total_CH4_out:.6f} mmol")
    # print(f"Total H2 at DFM outlet: {total_H2_out:.6f} mmol")
    # print(f"Total CO2 at DFM outlet: {total_CO2_out:.6f} mmol")
    # print(f"Total gas (CH4+CO2+H2) at DFM outlet: {total_hydrogenation_out:.6f} mmol")
    # print(f"Execution Time in seconds: {duration:.1f}")
    # print(f"Execution Time in minutes: {duration/60:.1f}")
        

    return total_CO2_in, total_CH4_out, total_H2_out, total_CO2_out 

# # # # ------------------------------ Model Params ------------------------------ #
# Time_ads = 51                      # Total simulation time (s) for adsorption                         #Var
# Time_prg = 133                        # Total simulation time (s) for purge                              #Var
# Time_hyd = 181                      # Total simulation time (s) for hydrogenation                       #Var
# Time_prg2 = 69                       # Total simulation time (s) for the 2nd purge                      #Var
# F_gas = 5                        # cm3/s                                                               #Var
# Di = 1.3        #1-10              # Reactor (Tube) Inside Diameter (cm)                               #Var
# T_K = 600            # Reactor temperature set (Isothermal mode) - in K

# total_CO2_in, total_CH4_out, total_H2_out, total_CO2_out = DFM_Model(688.7694971058909, 5.999999942427284, 2.8351749145988783)

# for k in [4]:
#     for j in [30]:
#         for i in [673]:
#             for l in [700]:
#                 print('=====================', flush=True)
#                 print(f"Diameter: {k} cm", flush=True)
#                 print(f"Temperature: {i} K", flush=True)
#                 print(f"Flowrate: {j} mL/s", flush=True)
#                 print(f"Hydrogenation time: {l} s", flush=True)
#                 print('=====================', flush=True)
#                 total_CO2_in, total_CH4_out, total_H2_out, total_CO2_out = DFM_Model(
#                       200, 200, l, 200, j, k, i
#                       )
#                 print(f"Results: CO2_in={total_CO2_in}, CH4_out={total_CH4_out}, H2_out={total_H2_out}, CO2_out={total_CO2_out}", flush=True)
    
    
    