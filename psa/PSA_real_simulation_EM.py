# pyAPEP package import
import pyapep.isofit as isofit
import pyapep.simsep as simsep

# Timing counter import
from time import perf_counter
import os

# Data treatment package import
import numpy as np
import pandas as pd
import openpyxl

# Data visualization package import
import matplotlib.pyplot as plt

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ISOTHERM_CSV = os.path.join(_MODULE_DIR, 'Casestudy3_Zeolite13X.csv')

def PSA_simulation(mH2):
    
    start_time = perf_counter()
    
    if isinstance(mH2, list) and len(mH2) == 1:
        mH2 = mH2[0]
    
    mH2 = min(1, mH2)
    mCH4 = 1 - mH2
    
    # Data import for isotherm
    Data = pd.read_csv(_ISOTHERM_CSV)
    
    # Pure isotherm definition
    P_CH4 = Data['Pressure_CH4 (bar)'].dropna().values
    q_CH4 = Data['Uptake_CH4 (mol/kg)'].dropna().values

    P_H2 = Data['Pressure_H2 (bar)'].dropna().values
    q_H2 = Data['Uptake_H2 (mol/kg)'].dropna().values

    CH4_iso, CH4_p, CH4_name, CH4_err = isofit.best_isomodel(P_CH4, q_CH4)
    H2_iso, H2_p, H2_name, H2_err = isofit.best_isomodel(P_H2, q_H2)

    # print(f"CH₄ Isotherm Model: {CH4_name}, Error: {CH4_err}")
    # print(f"H₂ Isotherm Model: {H2_name}, Error: {H2_err}")

    CH4_iso_ = lambda P,T: CH4_iso(P)
    H2_iso_ = lambda P,T: H2_iso(P)
    
    def MixIso(P, T):
        q1 = CH4_iso(P[0])
        q2 = H2_iso(P[1])
        return q1, q2
    
    # Define column
    N = 21
    L = 1.35
    A_cros = np.pi*0.15**2
    CR1 = simsep.column(L, A_cros, n_component=2, N_node = N) # components: CH₄ and H₂
    
    ### Sorbent prop
    # High CH₄ adsorption capacity ensures that more CH₄ is captured, leaving H₂ in the gas phase and increasing recovery.
    # High CH₄ selectivity over H₂ improves H₂ purity.
    # High CH₄ selectivity ensures that H₂ is not adsorbed, increasing recovery.
    # High CH₄ selectivity reduces CH₄ contamination in the product gas, increasing H₂ purity.
    
    # Lower void fraction increases contact between the gas and adsorbent, improving CH₄ adsorption and H₂ recovery.
    # Lower void fraction improves CH₄ capture, increasing H₂ purity.
    voidfrac = 0.37      # (m^3/m^3) a smaller void fraction increases resistance to gas flow, reducing velocities and flowrates
    # Smaller particle sizes and lower void fractions increase the contact area between the gas and adsorbent.
    # Improves CH₄ adsorption, increasing H₂ recovery.
    # Reduces CH₄ contamination, improving H₂ purity.
    D_particle = 12e-4   # (m) larger particles reduce the pressure drop, stabilizing flowrates
    rho = 1324           # (kg/m^3)
    CR1.adsorbent_info(MixIso, voidfrac, D_particle, rho)

    ### Gas prop
    Mmol = [0.016, 0.002] # kg/mol (CH4, H2)
    mu_visco = [11.86E-6, 8.76E-6,]   # (Pa sec)
    CR1.gas_prop_info(Mmol, mu_visco)

    ### Transfer prop
    # Higher k_MTC improves the rate of CH₄ adsorption, leaving more H₂ in the gas phase and increasing recovery.
    # Faster mass transfer reduces CH₄ contamination, improving H₂ purity.
    k_MTC  = [1E-2, 1E-2]     # m/sec

    a_surf = 1 #Volumatric specific surface area (m2/m3)
    D_disp = [1E-2, 1E-2]     # m^2/sec 
    CR1.mass_trans_info(k_MTC, a_surf, D_disp)

    # High Delta H_ads can cause significant temperature increases, reducing adsorption capacity and lowering recovery.
    # High Delta H_ads can reduce CH₄ adsorption efficiency, lowering H₂ purity.
    dH_ads = [20.856e3, 5e3]   # J/mol (CH4, H2)
    Cp_s = 900
    Cp_g = [35.8, 28.836]  # J/mol/K

    h_heat = 100            # J/m2/K/s
    CR1.thermal_info(dH_ads, Cp_s, Cp_g, h_heat)
    
    ##################
    ### ADSORPTION ###
    ##################
    
    ### Operating conditions
    P_inlet = 9 # High feed pressure can lead to inflated flowrates and concentrations.
    # Higher feed pressure increases the driving force for adsorption, improving the ability of the adsorbent to capture CH₄.
    # This ensures most H₂ remains in the gas phase, increasing recovery.
    # Higher feed pressure improves CH₄ adsorption, reducing CH₄ concentration in the product gas and increasing H₂ purity.
    # Tradeoff: Excessively high feed pressure can increase energy costs.
    P_outlet = 9
    # Lower outlet adsorption pressure during blowdown improves desorption of CH₄, regenerating the adsorbent for the next cycle.
    # However, very low outlet pressure can cause excessive H₂ loss, reducing recovery.
    # Lower outlet pressure can reduce CH₄ contamination in the product gas, improving H₂ purity.
    T_feed = 323
    y_feed = [mCH4, mH2] #[CH4, H2]

    # High difference between Cv_inlet and CV_outlet can result in high outlet velocities 
    # during adsorption, leading to inflated outlet volumetric flowrates, 
    # causing Overestimation inflating the molar flowrate of H₂
    
    # Higher Cv_inlet increases the inlet flowrate, potentially reducing the contact time between the gas and adsorbent, which can reduce CH₄ adsorption and lower recovery.
    # Higher Cv_inlet can reduce the residence time, leading to incomplete CH₄ adsorption and lower H₂ purity.
    Cv_inlet = 0.2E-1             # inlet valve constant (m/sec/bar)
    # Higher Cv_outlet increases the outlet flowrate, which can lead to H₂ loss and reduce recovery.
    # Higher Cv_outlet can reduce CH₄ contamination in the product gas, improving purity.
    Cv_outlet= 1E-1           # outlet valve constant (m/sec/bar)

    Q_feed = 0.05*A_cros  # volumetric flowrate (m^3/sec)

    CR1.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed,
                       Cv_inlet, Cv_outlet,
                       Q_inlet = Q_feed,
                       assigned_v_option = True,
                       foward_flow_direction = True)
    
    # Constants
    R = 8.3145  # Universal gas constant (J/mol/K)

    # Calculate total molar concentration
    C_total = P_inlet*100000 / (R * T_feed)
    # print(f"Total molar concentration (C_total): {C_total:.2f} mol/m³")

    # Calculate molar flowrate of CH4
    n_CH4_inlet = y_feed[0] * C_total * Q_feed
    # print(f"Molar flowrate of CH₄ at inlet: {n_CH4_inlet:.6f} mol/s")

    # Calculate molar flowrate of H2
    n_H2_inlet = y_feed[1] * C_total * Q_feed
    # print(f"Molar flowrate of H₂ at inlet: {n_H2_inlet:.6f} mol/s")

    # Calculate total molar flowrate
    n_total_inlet = n_CH4_inlet + n_H2_inlet
    # print(f"Total molar flowrate at inlet: {n_total_inlet:.6f} mol/s")
    
    ### Initial conditions
    P_init = 9.25*np.ones(N)    # (bar)
    y_init = [0.001*np.ones(N), 0.999*np.ones(N)] # (mol/mol)
    T_init = T_feed*np.ones(N)
    q_init = MixIso(P_init*np.array(y_init), T_init)

    CR1.initialC_info(P_init, T_init, T_init, y_init, q_init)
    # print(CR1)
    
    # Longer adsorption times allow more CH₄ to be adsorbed, leaving H₂ in the gas phase and improving recovery.
    # Longer adsorption times improve CH₄ capture, increasing H₂ purity
    y_res, z_res, t_res = CR1.run_mamoen(50,n_sec = 20, 
                                    CPUtime_print = False) #t_ads: 126

    # CR1.Q_valve(draw_graph=True, y=None)
    
    final_time_index = -1  # Last time step
    outlet_node_index = -1  # Last spatial node (z = L)
    
    # print("##### Adsorption Results #####")

    # CH4 gas concentration at inlet
    C_CH4_inlet_ads = y_res[final_time_index][0]  # CH4 concentration at intlet
    # print(f"CH4 concentration at intlet: {C_CH4_inlet_ads:.2f} mol/m³")

    # CH4 gas concentration at outlet
    C_CH4_outlet_ads = y_res[final_time_index][N + outlet_node_index]  # CH4 concentration at outlet
    # print(f"CH4 concentration at outlet: {C_CH4_outlet_ads:.2f} mol/m³")

    # H2 gas concentration at inlet
    C_H2_inlet_ads = y_res[final_time_index][N]  # H2 concentration at inlet
    # print(f"H2 concentration at intlet: {C_H2_inlet_ads:.2f} mol/m³")

    # H2 gas concentration at outlet
    C_H2_outlet_ads = y_res[final_time_index][2*N + outlet_node_index]  # H2 concentration at outlet
    # print(f"H2 concentration at outlet: {C_H2_outlet_ads:.2f} mol/m³")

    H2_purity_ads = C_H2_outlet_ads / (C_H2_outlet_ads + C_CH4_outlet_ads)
    # print(f"H2 Purity during adsorption: {H2_purity_ads * 100:.2f}%")
    
    # CH4 solid uptake at inlet
    q_CH4_inlet_ads = y_res[final_time_index][2*N] 
    # print(f"CH4 solid uptake at intlet: {q_CH4_inlet_ads:.2f} mol/kg")

    # CH4 solid uptake at outlet
    q_CH4_outlet_ads = y_res[final_time_index][3*N + outlet_node_index]
    # print(f"CH4 solid uptake at outlet: {q_CH4_outlet_ads:.2f} mol/kg")

    # H2 solid uptake at inlet
    q_H2_inlet_ads = y_res[final_time_index][3*N]
    # print(f"H2 solid uptake at intlet: {q_H2_inlet_ads:.2f} mol/kg")

    # H2 solid uptake at outlet
    q_H2_outlet_ads = y_res[final_time_index][4*N + outlet_node_index]
    # print(f"H2 solid uptake at outlet: {q_H2_outlet_ads:.2f} mol/kg")
    
    Q_feed_ads = CR1.Q_valve()[0][-1]  # Inlet volumetric flowrate at the final time step
    Q_outlet_ads = CR1.Q_valve()[1][-1]  # Outlet volumetric flowrate at the final time step

    # print(f"Feed volumetric flowrate: {Q_feed_ads:.6f} m³/s")
    # print(f"Outlet volumetric flowrate: {Q_outlet_ads:.6f} m³/s")

    n_H2_inlet_ads = C_H2_inlet_ads * Q_feed_ads
    # print(f"Molar flowrate of H₂ at inlet: {n_H2_inlet_ads:.6f} mol/s")
    
    n_total_gas_inlet_ads = n_H2_inlet_ads + (C_CH4_inlet_ads * Q_feed_ads)
    # print(f"Total molar flowrate at adsorption inlet: {n_total_gas_inlet_ads:.6f} mol/s")

    n_H2_outlet_ads = C_H2_outlet_ads * Q_outlet_ads
    # print(f"Molar flowrate of H₂ at outlet: {n_H2_outlet_ads:.6f} mol/s")

    H2_recovery_ads = n_H2_outlet_ads / n_H2_inlet_ads
    # print(f"H₂ Recovery during adsorption: {H2_recovery_ads * 100:.2f}%")
    
    # fig = CR1.Graph(11, 0, loc=[1,0.9], 
    #            yaxis_label = 'Gas concentration of CH4 (mol/m$^3$) during Adsorption',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR1.Graph(11, 2, loc=[1,0.9], 
    #            yaxis_label = 'Solid uptake of CH4 (mol/kg) during Adsorption',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR1.Graph(11, 1, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas concentration of H2 (mol/m$^3$) during Adsorption',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR1.Graph(11, 3, loc=[1.17,0.9], 
    #            yaxis_label = 'Solid uptake of H2 (mol/kg) during Adsorption',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR1.Graph(10, 4, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas Temperature during Adsorption (K)',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR1.Graph(10, 5, loc=[1.17,0.9], 
    #            yaxis_label = 'Solid Temperature during Adsorption (K)',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig, ax = CR1.Graph_P(10, loc=[1.15,0.9])
               
    ################
    ### BLOWDOWN ###
    ################
    
    import copy

    CR2 = copy.deepcopy(CR1)
    CR2.next_init()
    CR2.change_init_node(11)

    ### Operating conditions
    # Lower blowdown pressure improves CH₄ desorption, regenerating the adsorbent for the next cycle and increasing recovery.
    # Prevents CH₄ contamination in the product gas, improving H₂ purity.
    P_inlet = 9 # matching adsorption outlet pressure
    P_outlet = 1
    T_feed = 323
    y_feed = [0.001,0.999]

    # The inlet valve is closed, which is correct for blowdown
    Cv_inlet = 0E-1             # inlet valve constant (m/sec/bar)
    # The outlet valve constant is reasonable, 
    # but if the pressure difference is too large, it can result in high outlet flowrates.
    Cv_outlet= 1E-1           # outlet valve constant (m/sec/bar)

    CR2.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed,
                       Cv_inlet, Cv_outlet,
                       foward_flow_direction = False)
    # print(CR2)
    
    y_res = CR2.run_mamoen(100,n_sec = 10, 
                                    CPUtime_print = False)
    # CR2.Q_valve(draw_graph=True, y=None)
    
    # print("##### Blowdown Results #####")
    
    M=11

    final_time_index = -1  # Last time step
    outlet_node_index = -1  # Last spatial node (z = L)

    # CH4 gas concentration at inlet
    C_CH4_inlet_bd = y_res[0][final_time_index][0]  # CH4 concentration at intlet
    # print(f"CH4 concentration at intlet: {C_CH4_inlet_bd:.2f} mol/m³")

    # CH4 gas concentration at outlet
    C_CH4_outlet_bd = y_res[0][final_time_index][M + outlet_node_index]  # CH4 concentration at outlet
    # print(f"CH4 concentration at outlet: {C_CH4_outlet_bd:.2f} mol/m³")

    # H2 gas concentration at inlet
    C_H2_inlet_bd = y_res[0][final_time_index][M]  # H2 concentration at inlet
    # print(f"H2 concentration at intlet: {C_H2_inlet_bd:.2f} mol/m³")

    # H2 gas concentration at outlet
    C_H2_outlet_bd = y_res[0][final_time_index][2*M + outlet_node_index]  # H2 concentration at outlet
    # print(f"H2 concentration at outlet: {C_H2_outlet_bd:.2f} mol/m³")

    molar_ratio_blowdown = C_CH4_outlet_bd / C_H2_outlet_bd
    # print(f"Molar ratio of CH₄ to H₂ during blowdown: {molar_ratio_blowdown:.2f}")

    H2_purity_blowdown = C_H2_outlet_bd / (C_H2_outlet_bd + C_CH4_outlet_bd)
    # print(f"H₂ Purity during blowdown: {H2_purity_blowdown * 100:.2f}%")

    Q_feed_bd = CR2.Q_valve()[0][-1]  # Inlet volumetric flowrate at the final time step
    Q_outlet_bd = CR2.Q_valve()[1][-1]  # Outlet volumetric flowrate at the final time step
    
    # fig = CR2.Graph(10, 0, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas concentration of CH4 (mol/m$^3$) during blowdown',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR2.Graph(10, 2, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild uptake of CH4 (mol/kg) during blowdown',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR2.Graph(10, 1, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas concentration of H2 (mol/m$^3$) during blowdown',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR2.Graph(10, 3, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild uptake of H2 (mol/kg) during blowdown',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR2.Graph(10, 4, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas Temperature during blowdown (K)',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR2.Graph(10, 5, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild Temperature during blowdown (K)',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig, ax = CR2.Graph_P(10, loc=[1.15,0.9])
    
    #############
    ### Purge ###
    #############
    
    CR3 = copy.deepcopy(CR2)
    CR3.next_init()

    ### Operating conditions
    P_inlet = 1.5
    P_outlet = 1
    T_feed = 323
    y_feed = [0.001,0.999]

    # High difference between inlet and outlet valve constants can lead to
    # high outlet flowrates during the purge step, inflating the molar flowrate of H₂
    Cv_inlet = 1.0E-1             # inlet valve constant (m/sec/bar)
    Cv_outlet= 1.8E-1           # outlet valve constant (m/sec/bar)

    CR3.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed,
                       Cv_inlet, Cv_outlet,
                       foward_flow_direction = False)
    # print(CR3)
    
    # Insufficient purge can leave residual CH₄ adsorbed, reducing recovery in subsequent cycles.
    # Effective purging reduces CH₄ contamination, improving H₂ purity.
    y_res = CR3.run_mamoen(100,n_sec = 10, 
                                    CPUtime_print = False)
    # CR3.Q_valve(draw_graph=True, y=None)
    
    # print("##### Purge Results #####")
    
    M=11

    final_time_index = -1  # Last time step
    outlet_node_index = -1  # Last spatial node (z = L)

    # CH4 gas concentration at inlet
    C_CH4_inlet_pg = y_res[0][final_time_index][0]  # CH4 concentration at intlet
    # print(f"CH4 concentration at intlet: {C_CH4_inlet_pg:.2f} mol/m³")

    # CH4 gas concentration at outlet
    C_CH4_outlet_pg = y_res[0][final_time_index][M + outlet_node_index]  # CH4 concentration at outlet
    # print(f"CH4 concentration at outlet: {C_CH4_outlet_pg:.2f} mol/m³")

    # H2 gas concentration at inlet
    C_H2_inlet_pg = y_res[0][final_time_index][M]  # H2 concentration at inlet
    # print(f"H2 concentration at intlet: {C_H2_inlet_pg:.2f} mol/m³")

    # H2 gas concentration at outlet
    C_H2_outlet_pg = y_res[0][final_time_index][2*M + outlet_node_index]  # H2 concentration at outlet
    # print(f"H2 concentration at outlet: {C_H2_outlet_pg:.2f} mol/m³")

    H2_purity_purge = C_H2_outlet_pg / (C_H2_outlet_pg + C_CH4_outlet_pg)
    # print(f"H₂ Purity during purge: {H2_purity_purge * 100:.2f}%")

    Q_feed_pg = CR3.Q_valve()[0][-1]  # Inlet volumetric flowrate during purge
    Q_outlet_pg = CR3.Q_valve()[1][-1]  # Outlet volumetric flowrate during purge

    # Molar flowrate of H₂ at inlet
    n_H2_inlet_pg = C_H2_inlet_pg * Q_feed_pg
    # print(f"Molar flowrate of H₂ at inlet: {n_H2_inlet_pg:.6f} mol/s")

    # Molar flowrate of H₂ at outlet
    n_H2_outlet_pg = C_H2_outlet_pg * Q_outlet_pg
    # print(f"Molar flowrate of H₂ at outlet: {n_H2_outlet_pg:.6f} mol/s")

    # H₂ recovery during purge
    H2_recovery_purge = n_H2_outlet_pg / n_H2_inlet_pg
    # print(f"H₂ Recovery during purge: {H2_recovery_purge * 100:.2f}%")
    
    # fig = CR3.Graph(10, 0, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas concentration of CH4 (mol/m$^3$) during purge',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR3.Graph(10, 2, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild uptake of CH4 (mol/kg) during purge',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR3.Graph(10, 1, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas concentration of H2 (mol/m$^3$) during purge',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR3.Graph(10, 3, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild uptake of H2 (mol/kg) during purge',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig = CR3.Graph(10, 4, loc=[1.17,0.9], 
    #            yaxis_label = 'Gas Temperature during purge (K)',)
    #            # file_name = 'H2_gas_conc_test.png')

    # fig = CR3.Graph(10, 5, loc=[1.17,0.9], 
    #            yaxis_label = 'Soild Temperature during purge (K)',)
    #            # file_name = 'CH4_gas_conc_test.png')
               
    # fig, ax = CR3.Graph_P(10, loc=[1.15,0.9])
    
    ######################
    ### Pressurisation ###
    ######################
    
    N = 11
    R_gas = 8.3145      # 8.3145 J/mol/K

    total_y = []
    CR4 = copy.deepcopy(CR3)
    CR4.next_init()
    P_outlet = 1
    P_inlet = 2
    while P_outlet<9.1:
        ### Operating conditions
        T_feed = 323
        y_feed = [mCH4, mH2]

        Cv_inlet =(P_inlet-P_outlet)*0.1             # inlet valve constant (m/sec/bar)
        # The outlet valve is closed, which is correct for pressurization
        Cv_outlet= 0E-1           # outlet valve constant (m/sec/bar)

        CR4.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed,
                           Cv_inlet, Cv_outlet,
                           foward_flow_direction = True)
        y_res = CR4.run_mamoen(5,n_sec = 10, 
                                    CPUtime_print = False)
        total_y.append(y_res)
    
        P = np.zeros(N)
        for ii in range(2):
            Tg_res = y_res[0][:,2*2*N : 2*2*N+N]
            P = P + y_res[0][:,(ii)*N:(ii+1)*N]*R_gas*Tg_res/1E5
        
        P_outlet = np.mean(P[-1])
        P_inlet = P_outlet+1.1
        # fig, ax = CR4.Graph_P(1, loc=[1.15,0.9])
        # plt.show()
        CR4.next_init()
    
        # CR4.Q_valve(draw_graph=True, y=None)
        
        for ii, _y in enumerate(total_y):
            if ii == 0:
                concat_y = _y[0]
            else:
                concat_y = np.concatenate([concat_y, _y[0]], axis=0)
        
        # Index values for graph plotting
        # 0. CH₄ Gas Concentration: Columns ( 0 ) to ( N-1 ) (( 0 ) to ( 10 )).
        # 1. H₂ Gas Concentration: Columns ( N ) to ( 2N-1 ) (( 11 ) to ( 21 )).
        # 2. CH₄ Solid Uptake: Columns ( 2N ) to ( 3N-1 ) (( 22 ) to ( 32 )).
        # 3. H₂ Solid Uptake: Columns ( 3N ) to ( 4N-1 ) (( 33 ) to ( 43 )).
        # 4. Gas Temperature: Columns ( 4N ) to ( 5N-1 ) (( 44 ) to ( 54 )).
        # 5. Solid Temperature: Columns ( 5N ) to ( 6N-1 ) (( 55 ) to ( 65 )).

        # index=0
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Gas concentration H2 (mol/m$^3$) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # index=1
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Gas concentration CH4 (mol/m$^3$) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # index=2
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Soild uptake of CH4 (mol/kg) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # index=3
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Soild uptake of H2 (mol/kg) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # index=4
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Gas Temperature (K) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # index=5
        # one_sec = 10
        # every_n_sec = 10
        # n_show = one_sec*every_n_sec
        # z_dom= np.linspace(0,L,N)

        # n_sec = 10
        # t_max_int = np.int32(np.floor(len(concat_y)/5), )
        # n_t = t_max_int*n_sec+ 1
        # t_dom = np.linspace(0,t_max_int, n_t)

        # lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']

        # cc= 0
        # fig, ax = plt.subplots(figsize = [7,5], dpi = 90)
        # for j in range(0,len(concat_y), n_show):
        #     if j <= 1:
        #         lcolor = 'r'
        #     elif j >= len(concat_y)-n_show:
        #         lcolor = 'b'
        #     else:
        #         lcolor = 'k'
        #     ax.plot(z_dom,concat_y[j, index*N:(index+1)*N],
        #     color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        #     label = 't = {}'.format(t_dom[j]))
        #     cc = cc + 1
    
        # fig.legend(fontsize = 14,bbox_to_anchor =[1.17,0.9])
        # ax.set_xlabel('z-domain (m)', fontsize = 15)
        # plt.ylabel('Solid Temperature (K) during pressurisation', fontsize=15)
        # plt.xticks(fontsize = 13)
        # plt.yticks(fontsize = 13)
        # plt.grid(linestyle = ':')
        
        # print("##### Process Results #####")
        
        # Adsorption step
        n_H2_outlet_ads = C_H2_outlet_ads * Q_outlet_ads
        n_CH4_outlet_ads = C_CH4_outlet_ads * Q_outlet_ads

        # Blowdown step
        n_H2_outlet_bd = C_H2_outlet_bd * Q_outlet_bd
        n_CH4_outlet_bd = C_CH4_outlet_bd * Q_outlet_bd

        # Purge step
        n_H2_outlet_pg = C_H2_outlet_pg * Q_outlet_pg
        n_CH4_outlet_pg = C_CH4_outlet_pg * Q_outlet_pg

        n_H2_total = n_H2_outlet_ads + n_H2_outlet_bd + n_H2_outlet_pg
        n_CH4_total = n_CH4_outlet_ads + n_CH4_outlet_bd + n_CH4_outlet_pg

        n_total_outlet = n_H2_total + n_CH4_total
        # print(f"Total molar flowrate recovered at outlets: {n_total_outlet:.6f} mol/s")

        process_purity_H2 = n_H2_total / (n_H2_total + n_CH4_total)
        # print(f"Process Purity (H₂): {process_purity_H2 * 100:.8f}%")

        n_H2_feed = n_H2_inlet_ads # n_H2_inlet # C_H2_inlet_ads * Q_feed_ads
        # print(f"Molar flowrate of H₂ in feed: {n_H2_feed:.6f} mol/s")

        process_recovery_H2 = n_H2_total / n_H2_feed
        # print(f"Process Recovery (H₂): {process_recovery_H2 * 100:.8f}%")
        
        # print(f"Molar flowrate of H₂ during adsorption: {n_H2_outlet_ads:.6f} mol/s")
        # print(f"Molar flowrate of H₂ during blowdown: {n_H2_outlet_bd:.6f} mol/s")
        # print(f"Molar flowrate of H₂ during purge: {n_H2_outlet_pg:.6f} mol/s")
        # print(f"Total molar flowrate of H₂ exiting the column: {n_H2_total:.6f} mol/s")
        # print(f"Total molar flowrate of CH₄ exiting the column: {n_CH4_total:.6f} mol/s")
        # print(f"Molar flowrate of H₂ in feed: {n_H2_feed:.6f} mol/s")
        # print("Outlet volumetric flowrates:")
        # print(f"Adsorption: {Q_outlet_ads:.6f} m³/s")
        # print(f"Blowdown: {Q_outlet_bd:.6f} m³/s")
        # print(f"Purge: {Q_outlet_pg:.6f} m³/s")
        # print("Outlet gas concentrations:")
        # print(f"Adsorption H₂: {C_H2_outlet_ads:.2f} mol/m³, CH₄: {C_CH4_outlet_ads:.2f} mol/m³")
        # print(f"Blowdown H₂: {C_H2_outlet_bd:.2f} mol/m³, CH₄: {C_CH4_outlet_bd:.2f} mol/m³")
        # print(f"Purge H₂: {C_H2_outlet_pg:.2f} mol/m³, CH₄: {C_CH4_outlet_pg:.2f} mol/m³")
        
        duration = perf_counter() - start_time
        # print(f"Execution Time in seconds: {duration:.1f}")
        
        return n_H2_feed, n_total_gas_inlet_ads, n_H2_total, n_CH4_total
   
# n_H2_feed, n_total_gas_inlet_ads, n_H2_total, n_CH4_total = PSA_simulation(0.966863224)
# print(f"Molar flowrate of H₂ in feed: {n_H2_feed:.6f} mol/s")
# print(f"Total molar flowrate at adsorption inlet: {n_total_gas_inlet_ads:.6f} mol/s")
# print(f"Total molar flowrate of H₂ exiting the column: {n_H2_total:.6f} mol/s")
# print(f"Total molar flowrate of CH₄ exiting the column: {n_CH4_total:.6f} mol/s")

