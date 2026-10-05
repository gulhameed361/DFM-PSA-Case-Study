import numpy as np

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



# This function concerts mmol/mL to % for the reactor outlet temperature
def convert_mmol_per_mL_to_percent(C_mmol_per_mL, T, P):
    """
    Convert concentration from mmol/mL to volume %.

    Parameters:
    - C_mmol_per_mL: Array or scalar concentration in mmol/mL
    - T: Temperature (K)
    - P: Pressure (atm)

    Returns:
    - C_percent: Converted concentration in %
    """
    R = 0.08206  # Gas constant in L·atm / mol·K
    C_percent = (C_mmol_per_mL * R * T / P) * 100
    return C_percent

tau_increase = 20
tau_decrease = 20

def apply_minor_analyser_delay(C_model, dt, tau_increase, tau_decrease):
    Nt = len(C_model)
    C_measured = np.zeros(Nt)
    C_measured[0] = C_model[0]  # Initial condition

    for t in range(1, Nt):  # Iterate up to Nt only
        if C_model[t] > C_model[t - 1]:  # Concentration is increasing
            tau = tau_increase
        else:  # Concentration is decreasing
            tau = tau_decrease

        # Apply the delay based on the respective time constant
        dC_measured_dt = (C_model[t - 1] - C_measured[t - 1]) / tau
        C_measured[t] = C_measured[t - 1] + dC_measured_dt * dt

    # Ensure output length is exactly Nt
    C_measured = C_measured[:Nt]

    return C_measured


def apply_major_analyzer_delay(C_model, dt, tau_rise, zeta_rise, tau_fall, zeta_fall):
    """
    Applies a 2nd-order response model to smooth the CH4 concentration profile, 
    dynamically adjusting tau and zeta based on the analyzer output trend.
    
    Parameters:
    - C_model: Array of model-predicted CH4 concentration over time
    - dt: Time step size (s)
    - tau_rise, zeta_rise: Time constant and damping factor for the increasing phase
    - tau_fall, zeta_fall: Time constant and damping factor for the decreasing phase
    
    Returns:
    - C_measured: Smoothed concentration profile observed by the analyzer
    """
    Nt = len(C_model)
    C_measured = np.zeros(Nt)
    C_measured[0] = C_model[0]  # Initial condition
    C_dot = np.zeros(Nt)  # First derivative (rate of change)

    for t in range(1, Nt):
        # Check if the analyzer output is still increasing or has started decreasing
        if C_measured[t-1] >= C_measured[t-2]:  # Still rising
            tau, zeta = tau_rise, zeta_rise
        else:  # Transition to decreasing phase
            tau, zeta = tau_fall, zeta_fall

        # Compute second-order smoothing
        dC_dot_dt = (C_model[t] - C_measured[t-1] - 2*zeta*tau*C_dot[t-1]) / (tau**2)
        C_dot[t] = C_dot[t-1] + dC_dot_dt * dt
        C_measured[t] = C_measured[t-1] + C_dot[t] * dt

    return C_measured