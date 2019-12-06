import numpy as np
import pandas as pd
from atmosp import calculate as ac


# saturation vapour pressure [hPa]
def cal_es_hPa_1(Temp_C, Press_hPa):
    Press_kPa = Press_hPa/10
    e_mb = 6.1121*np.exp(((18.678 - Temp_C/234.5)*Temp_C)/(Temp_C + 257.14))
    f = 1.00072 + Press_kPa*(3.2E-6 + 5.9E-10*Temp_C**2)
    es_hPa = e_mb*f
    return es_hPa


def cal_es_hPa_2(Temp_C, Press_hPa):
    Press_kPa = Press_hPa/10
    e_mb = 6.1115*np.exp(((23.036 - Temp_C/333.7)*Temp_C)/(Temp_C + 279.82))
    f = 1.00022 + Press_kPa*(3.83E-6 + 6.4E-10*Temp_C**2)
    es_hPa = e_mb*f
    return es_hPa


def cal_es_hPa_3(Temp_C, Press_hPa):
    Press_kPa = Press_hPa/10
    Temp_C = 0.001
    e_mb = 6.1121*np.exp(((18.678 - Temp_C/234.5)*Temp_C)/(Temp_C + 257.14))
    f = 1.00072 + Press_kPa*(3.2E-6 + 5.9E-10*Temp_C**2)
    es_hPa = e_mb*f
    return es_hPa


def cal_vap_sat(Temp_C, Press_hPa):
    es_hPa_1 = cal_es_hPa_1(Temp_C, Press_hPa)[
        (Temp_C >= 0.00100) & (Temp_C < 50)]
    es_hPa_2 = cal_es_hPa_2(Temp_C, Press_hPa)[
        (Temp_C >= -40) & (Temp_C < -.001)]
    es_hPa_3 = cal_es_hPa_3(Temp_C, Press_hPa)[
        (Temp_C >= -0.001) & (Temp_C < .001)]
    es_hPa = pd.concat([es_hPa_1, es_hPa_2, es_hPa_3],
                       sort=True).sort_values(axis='index')
    return es_hPa


# density of dry air [kg m-3]
def cal_dens_dry(RH_pct, Temp_C, Press_hPa):
    gas_ct_dry = 8.31451/0.028965  # dry_gas/molar
    es_hPa = cal_vap_sat(Temp_C, Press_hPa)
    Ea_hPa = RH_pct/100*es_hPa
    dens_dry = ((Press_hPa - Ea_hPa)*100)/(gas_ct_dry*(273.16 + Temp_C))
    return dens_dry


# density of vapour [kg m-3]
def cal_dens_vap(RH_pct, Temp_C, Press_hPa):
    gas_ct_wv = 8.31451/0.0180153  # dry_gas/molar_wat_vap
    es_hPa = cal_vap_sat(Temp_C, Press_hPa)
    Ea_hPa = RH_pct/100*es_hPa
    vap_dens = (Ea_hPa*100/((Temp_C + 273.16)*gas_ct_wv))
    return vap_dens


# specific heat capacity of air mass [J kg-1 K-1]
def cal_cpa(Temp_C, RH_pct, Press_hPa):
    # heat capacity of dry air depending on air temperature
    cpd = 1005.0 + ((Temp_C + 23.16)**2)/3364.0
    # heat capacity of vapour
    cpm = 1859 + 0.13*RH_pct + (19.3 + 0.569*RH_pct) * \
        (Temp_C/100.) + (10.+0.5*RH_pct)*(Temp_C/100.)**2

    # density of dry air
    rho_d = cal_dens_dry(RH_pct, Temp_C, Press_hPa)

    # density of vapour
    rho_v = cal_dens_vap(RH_pct, Temp_C, Press_hPa)

    # specific heat
    cpa = cpd*(rho_d/(rho_d + rho_v)) + cpm*(rho_v/(rho_d + rho_v))
    return cpa


# air density [kg m-3]
def cal_dens_air(Press_hPa, Temp_C):
     # dry_gas/molar
    gas_ct_dry = 8.31451/0.028965

    # air density [kg m-3]
    dens_air = (Press_hPa*100)/(gas_ct_dry*(Temp_C + 273.16))
    return dens_air


# Obukhov length
def cal_Lob(QH, UStar, Temp_C, RH_pct, Press_hPa, g=9.8, k=0.4):
    # gravity constant/(Temperature*Von Karman Constant)
    G_T_K = (g/(Temp_C + 273.16))*k

    # air density [kg m-3]
    rho = cal_dens_air(Press_hPa, Temp_C)

    # specific heat capacity of air mass [J kg-1 K-1]
    cpa = cal_cpa(Temp_C, RH_pct, Press_hPa)

    # Kinematic sensible heat flux [K m s-1]
    H = QH/(rho*cpa)

    # friction velocity
    uStar = UStar.where(UStar > 0.01, 0.01)
    # temperature scale
    TStar = -H/uStar

    # Obukhov length
    Lob = (uStar**2)/(G_T_K*TStar)

    return Lob


# Calculate slope of es(Ta), i.e., saturation evaporation pressure `es` as function of air temperature `ta [K]`
def cal_des_dta(Temp_C, Press_hPa, dta=1.0):
    # air temperature [K]
    ta = Temp_C + 273.16

    # atmospheric pressure [Pa]
    pa = Press_hPa*100

    # change in saturation pressure [Pa]
    des = ac('es', p=pa, T=ta + dta/2) - ac('es', p=pa, T=ta - dta/2)

    # slope in saturation pressure curve at `ta` [Pa K-1]
    des_dta = des/dta

    return pd.Series(des_dta, index=Temp_C.index)


# Calculating vapour pressure deficit
def cal_vpd(Temp_C, RH_pct, Press_hPa):

    # air temperature [K]
    ta = Temp_C + 273.16

    # atmospheric pressure [Pa]
    pa = Press_hPa*100

    # relative humidity [%]
    rh = RH_pct

    # actual vapour pressure [Pa]
    e = ac('e', p=pa, T=ta, RH=rh)

    # saturation vapour pressure [Pa]
    es = ac('es', p=pa, T=ta)

    # vapour pressure deficit [Pa]
    vpd = es-e

    return pd.Series(vpd, index=Temp_C.index)


# Calculating psychrometric constant
def cal_gamma(Press_hPa):

    # atmospheric pressure [Pa]
    pa = Press_hPa*100

    # psychrometric constant [hPa K-1]
    gamma = 0.665e-3 * pa

    return gamma
