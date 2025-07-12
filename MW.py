import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
 
R = 289.32                                                    # air gas constant
print("\nR: ", R)
gamma = 1.4                                                   # heat capacity (dry air at room temperature)
real_data = True

modes = ['no_source', 'daylight_sine', 'fullday_sine']        # Q(t) type of function
mode = modes[1]
boundaries = ['neighbor', 'const_gradient', 'distribution']   # boundary conditions type on x
boundary = boundaries[0]

interpol_steps_x = 10                                         # for interpolation (number of steps between cities)
d_t_fixed = 1                                                 # for interpolation [s]
Q_max = 0.45                                                  # solar power absorbed per unit volume Q(t) (J/(sm^3)) or energy source density (50 W/m^2) # 111 m
idx_K = 13                                                    # if Klaipeda is 0, Kaunas is 13

# conditions for Q(t)
day_start = 3*3600                 # cut times for Q(t)
day_end = 16*3600                  # cut times for Q(t)

smooth = True
print_steps = False

# interpolation function
def interpol(values, n_steps):
    """ Interpolates linearly between array values, with n_steps interpolated points (excluding startpoint) per pair. """
    if not smooth:
        interpolated = []
        for i in range(len(values) - 1):
            segment = np.linspace(values[i], values[i + 1], int(n_steps) + 1)[:-1]
            interpolated.extend(segment)
        interpolated.append(values[-1])
        return np.array(interpolated)
    
    else:
        n_steps = int(n_steps)
        x = np.arange(len(values))
        cs = CubicSpline(x, values)
        x_new = np.linspace(0, len(values) - 1, (len(values) - 1) * n_steps + 1)
        return cs(x_new)

def initial_data(interpol_steps_x = 10):
    # get real d_x (Kaunas: 195 km in a straight line from Klaipeda to Vilnius)
    dists_map = np.array([0, 32000, 47000, 59000, 68000, 77000, 86000, 102000, 107000, 122000, 134000, 155000, 170000, 195000, 200000, 210000, 215000, 228000, 238000, 255000, 267000, 278000, 286000])
    d_x_cities_map = [dists_map[i] - dists_map[i-1] for i in (1, len(dists_map) - 1)]
    d_x_map = 10 # d_x_cities_map / interpol_steps_x
    print("size of dx_cities_map: ", d_x_cities_map,
          "\nsize of dx_map: ", d_x_map)
    # space_x = np.linspace(dists_map[0], dists_map[-1], N)
    # if I vary my distances, I should be careful with my scheme

    # initial values (23 cities)
    # rho_0 = np.array([1.221851938, 1.2100785689443974, 1.219594327344099, 1.2145004488123767, 1.2152435428841997, 1.2071443789039433, 1.2144920931547245, 1.2022834963404432, 1.2024453879111674, 1.2195846277898634, 1.2144920931547245, 1.2129911765699595, 1.2152377167088704, 1.2186806178125595, 1.1952159580124555, 1.2145171203258096, 1.2153892280270455, 1.2191815547654303, 1.209284468697123, 1.2024949748865055, 1.2078789541693364, 1.2081772513393536, 1.2081705508529765])
    rho_0 = np.array([1.246490633,1.268956931,1.269893025,1.235827324,1.285134211,1.267777285,1.251549069,1.26570525,1.251549069,1.251549069,1.24944421,1.245981379,1.257693976,1.250193622,1.251549069,1.266193872,1.251549069,1.221425829,1.266113085,1.199768348,1.252184522,1.263164892,1.238322027])
    m_0 = np.array([15.84028402, 9.088019194143177, 17.17403461502104, 10.207089275977195, 6.845107164978002, 1.4570329224920908, 5.700728725900825, 2.149875256816127, 5.644182455223987, 5.724632676075396, 4.940631562447381, 2.3316989749674892, 1.7927575672958602, 1.5254371534833944, 5.610248089633825, 6.786721668380624, 5.70493980522071, 4.959708592405275, 4.5950874955340755, 2.4190350910796856, 5.183713993649857, 3.240621350632468, 9.18170957190635]) * np.cos(3 * np.pi / 8)
    E_0 = np.array([251880.7745, 256299.18999804466, 255962.68481684025, 248773.23748558539, 258570.15473125436, 256435.26207519686, 251706.81613229058, 252711.27591046967, 251706.68342045517, 251706.87223390548, 251280.1874026983, 250425.78157982603, 252879.9956166658, 251271.0927039952, 251706.6037778565, 255268.10835034147, 251706.82601552532, 253989.33820598767, 254834.57779863456, 248224.82016225604, 251027.28192201792, 253221.6583121058, 248172.61627729976])
    P_0 = np.array([100711.2386, 102506.0253, 102336.7058, 99492.13819999999, 103420.35059999999, 102573.75309999999, 100677.3747, 101083.7415, 100677.3747, 100677.3747, 100508.0552, 100169.41619999999, 101151.4693, 100508.0552, 100677.3747, 102099.65849999999, 100677.3747, 101591.7, 101930.339, 99288.95479999999, 100406.4635, 101286.9249, 99255.0909])
    total_dist = 286000                        # 286 km
    d_x_cities = total_dist / (len(rho_0) - 1) # 13 km
    print("initial data: len(rho_0) before interpolation: ", len(rho_0))

    # interpolation on initial values
    rho_0 = interpol(rho_0, interpol_steps_x)
    m_0   = interpol(m_0, interpol_steps_x)
    E_0   = interpol(E_0, interpol_steps_x)
    P_0   = interpol(P_0, interpol_steps_x)

    # space on x
    d_x = d_x_cities / interpol_steps_x # distance between each x point after interpolation [m]
    N = len(rho_0)
    b = 0
    c = b + N * d_x
    space_x = np.linspace(b, c, N)

    # Kaunas
    x_K = idx_K * interpol_steps_x # number of steps to reach Kaunas
    print("initial data: len(rho_0) after interpolation: ", len(rho_0))
    print("distance Klaipeda - Kaunas", x_K * d_x)

    print("\n rho_0 for Kaunas: ", rho_0[x_K])
    return rho_0, m_0, E_0, P_0, space_x, d_x, b, c, x_K, d_x_map

def boudaries_data(interpol_steps_t = 60 * 3):
    # Klapeda (check)
    rho_b = np.array([1.246490633,1.246986778,1.245994882,1.245575919,1.245994882,1.252891699,1.255823129,1.258097029,1.25860161,1.260287751,1.262570263,1.264348034,1.263671073,1.263586697,1.2592088,1.25988178,1.261647486,1.262658685,1.262658685,1.261730899,1.260634437,1.264348034,1.262486262,1.253414159,1.250812564])
    # rho_b = np.array([1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765])
    m_b = np.array([15.66291633,
            9.127699515601016, 9.127699515601016, 10.153890585402314, 6.643236925555771,
            1.7823318580759382, 1.1342111824119607, 1.5662916328546124, 1.9443620269919328,
            1.3502514076332865, 2.5384726463505785, 3.4026335472358817, 4.212784391815854,
            4.2667944481211855, 5.292985517922483, 3.8887240539838657, 2.322432421129253,
            2.322432421129253, 1.5662916328546124, 1.1342111824119607, 2.700502815266573,
            1.5662916328546124, 1.6743117454652754, 6.697246981861102, 6.481206756639776]) * np.cos(3 * np.pi / 8)
    E_b = np.array([251880.7745, 251812.96672618669, 251812.96672618669, 251736.58836494145, 251796.56754975242,
            251864.08581396198, 251947.95441846398, 252202.4220298825, 252202.97754099607,
            252286.81806471647, 252542.73122593388, 252798.85926617542, 252970.7609771759,
            253055.61240903268, 253144.37805765896, 253223.6414139842, 253219.56970065722,
            253219.56970065722, 253218.3390298825, 253133.19091846395, 252966.38525886586,
            252795.04027988252, 252625.86728830807, 252389.48736292834, 251964.99701106717])
    P_b = np.array([100711.2386,
            100711.2386, 100711.2386, 100677.3747, 100711.2386, 100745.1025, 100778.9664,
            100880.5581, 100880.5581, 100914.42199999999, 101016.0137, 101117.6054,
            101185.3332, 101219.1971, 101253.06099999999, 101286.9249, 101286.9249,
            101286.9249, 101286.9249, 101253.06099999999, 101185.3332, 101117.6054,
            101049.87759999999, 100948.28589999999, 100778.9664]) 

    # Vilnius
    rho_c = np.array([1.238322027,1.238867972,1.238942514,1.238768052,1.241187563,1.241513272,1.241338907,1.240588959,1.238917206,1.237921031,1.237174951,1.237101149,1.237174951,1.237524234,1.237524234,1.237450322,1.236705572,1.236558611,1.235320492,1.234404209,1.233344651,1.232533761,1.22752065,1.23527652,1.236018027]) + 0.011871595
    # rho_c = np.array([1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765,
    #         1.2081705508529765, 1.2081705508529765, 1.2081705508529765, 1.2081705508529765])
    m_c = np.array([9.181709572, 8.479578839937039, 9.99186041648632, 10.315920754318311, 12.422312950226235,
            10.045870472791652, 9.397749797127673, 9.721810134959664, 10.045870472791652,
            9.451759853433005, 9.127699515601016, 8.641609008853035, 10.315920754318311,
            10.531960979539637, 9.613790022349, 7.399377713830409, 9.505769909738337,
            7.453387770135742, 9.019679402990354, 9.343739740822343, 6.535216812945107,
            6.859277150777095, 9.073689459295686, 6.373186644029113, 5.130955349006489]) * np.cos(3 * np.pi / 8)
    E_c = np.array([248172.6163, 247828.8453307582, 247755.74604184547, 247673.80980899386, 247778.29111398463,
            247671.5341752063, 247581.659179603, 247584.2233420246, 247502.21467520628,
            247497.4207538427, 247494.92902538783, 247576.01415913057, 247504.49030899385,
            247675.6737764039, 247668.0187133552, 247737.0870520204, 247751.823742548,
            247906.73854103455, 247917.4165149607, 247835.21951982885, 247901.42309511063,
            247987.87920748504, 247240.54293794144, 248239.19650920838, 248233.28227582635])
    P_c = np.array([99255.0909, 99119.6353, 99085.7714, 99051.9075, 99085.7714, 99051.9075, 99018.04359999999,
            99018.04359999999, 98984.1797, 98984.1797, 98984.1797, 99018.04359999999,
            98984.1797, 99051.9075, 99051.9075, 99085.7714, 99085.7714, 99153.4992,
            99153.4992, 99119.6353, 99153.4992, 99187.36309999999, 98882.58799999999,
            99288.95479999999, 99288.95479999999]) 

    print("boundary x=b: len(rho_b) before interpolation: ", len(rho_b))
    rho_b = interpol(rho_b, interpol_steps_t)
    m_b   = interpol(m_b, interpol_steps_t)
    E_b   = interpol(E_b, interpol_steps_t)
    P_b   = interpol(P_b, interpol_steps_t)
    print("boundary x=b: len(rho_b) after interpolation: ", len(rho_b))

    rho_c = interpol(rho_c, interpol_steps_t)
    m_c   = interpol(m_c, interpol_steps_t)
    E_c   = interpol(E_c, interpol_steps_t)
    P_c   = interpol(P_c, interpol_steps_t)

    U_b = [
        {"rho": rho_b[i], "m": m_b[i], "E": E_b[i]} for i in range(len(rho_b))
    ]

    U_c = [
        {"rho": rho_c[i], "m": m_c[i], "E": E_c[i]} for i in range(len(rho_c))
    ]

    return U_b, U_c

def Kaunas_data(interpol_steps_t = 60 * 3):
    ''' Temperature is Celsius. '''
    # rho_0(Kaunas) = 1.2186806178125595
    # rho_K = np.array([1.298, 1.299, 1.3, 1.302, 1.303, 1.305, 1.306, 1.305, 1.301, 1.296, 1.291, 1.287, 1.283, 1.281, 1.284, 1.288, 1.292, 1.296, 1.299, 1.302, 1.304, 1.304, 1.303, 1.301, 1.300]) - 0.07931938218 # added the last value, and shift to match
    rho_K = np.array([1.250193622,1.245999476,1.245999476,1.245999476,1.250996767,1.250996767,1.244737065,1.244737065,1.244737065,1.245999476,1.245999476,1.245999476,1.242299339,1.242299339,1.243556727,1.243556727,1.249786708,1.251049119,1.256066662,1.256066662,1.256066662,1.256066662,1.256066662,1.256066662,1.259852037])
    m_K = np.array([1.525437153,9.806381701,10.895979668,9.806381701,9.261582718,9.261582718,10.895979668,11.985577635,11.440778651,11.985577635,10.895979668,10.895979668,9.806381701,10.895979668,6.537587801,5.447989834,7.627185767,5.447989834,5.447989834,4.90319085,5.447989834,4.90319085,4.90319085,1.63439695,3.813592884])* np.cos(3 * np.pi / 8)
    E_K = np.array([251271.0927,250716.9744,250726.2291,250716.9744,250712.7123,250712.7123,250472.2499,250482.4789,250477.2426,250736.4581,250726.2291,250726.2291,250970.9536,250980.2084,251203.0136,251197.6556,251463.3251,251705.6141,251705.6141,251703.3004,251705.6141,251703.3004,251703.3004,251440.5535,251445.4244])
    P_K = np.array([100508.0552,100271.0079,100271.0079,100271.0079,100271.0079,100271.0079,100271.0079,100271.0079,100271.0079,100169.4162,100169.4162,100169.4162,100169.4162,100169.4162,100372.5996,100372.5996,100372.5996,100372.5996,100474.1913,100474.1913,100575.783,100575.783,100677.3747,100575.783,100575.783])

    rho_K = interpol(rho_K, interpol_steps_t)
    m_K   = interpol(m_K, interpol_steps_t)
    E_K   = interpol(E_K, interpol_steps_t)
    P_K   = interpol(P_K, interpol_steps_t)

    U_K = [
        { "rho": rho_K[i], "m": m_K[i], "E": E_K[i], "P": P_K[i]}
        for i in range(len(rho_K))
    ]

    print("\n m_K at t=0: ", m_K[0])
    return U_K

def Kaunas_temp(interpol_steps_t = 60 * 3):
    # T_K_real = np.array([4.722222222,5,5,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,3.888888889,6.111111111,6.111111111,6.111111111,6.111111111,6.111111111,6.111111111,6.111111111,6.111111111,2.777777778,3.888888889])
    T_K_real = np.array([4.3,4.2,4.3,4.4,4.7,4.8,4.9,5.1,5.4,5.8,5.9,6,5.7,5.5,4.7,4.2,4.3,4.4,4.2,3.5,4.1,5,4.5,3.8,6])
    T_K_real = interpol(T_K_real, interpol_steps_t)
    return T_K_real

if real_data == True:
    rho_0, m_0, E_0, P_0, space_x, d_x, b, c, x_K, d_x_map = initial_data(interpol_steps_x = interpol_steps_x) # initial values

    interpol_steps_t = 60 * np.floor(60 / d_t_fixed)  # number of t_steps in one hour
    M = 24 * interpol_steps_t                         # number of t_steps in one day
    print("total number of steps in a day: ", M)

    U_b, U_c = boudaries_data(interpol_steps_t = interpol_steps_t) # time series values
    U_K = Kaunas_data(interpol_steps_t = interpol_steps_t) # time series values
    T_K_real = Kaunas_temp(interpol_steps_t = interpol_steps_t) # time series values

elif real_data == False:
    d_t_fixed = 20

    ########## Step Size, Boundary Conditions, Number of Steps
    d_x = 10000        # x step size 
    b = 0              # initial x
    c = 285000         # final x

    M = 50                             # number of time steps [t]
    N = int(np.floor((c - b) / d_x))   # number of space steps [x]
    space_x = np.linspace(b, c, N)     # x grid [m]

    day_start = 0*3600                 # cut times for Q(t)
    day_end = 1371                     # cut times for Q(t)
    Q_max = 100                        # solar power absorbed per unit volume Q(t) (J/(sm^3)) or energy source density

    ########## Initial Values (Parameters)

    vel = 5                                                      # velocity [m/s]

    var = 0.001 * (c - b) ** 2
    mean = (c - b) / 2
    P_0 = 95000 + 10000*np.exp(-((space_x - mean) ** 2) / (2 * var)) / 10 # pressure [Pa] # np.ones(N)*100000 # np.random.uniform(95000, 105000, N)
    # plt.plot(space_x, P_0)
    # plt.show()
    rho_0 = 1 + 0.5*np.exp(-((space_x - mean) ** 2) / (2 * var)) / 10 # density [kg/m^3] # np.ones(N) # np.random.uniform(1.0, 1.5, N)
    # plt.plot(space_x, rho_0)
    # plt.show()

    m_0 = rho_0 * vel                               # air momentum density [kg/(s*m^2)]
    E_0 = P_0 / (gamma - 1) + 0.5 * m_0**2 / rho_0  # total energy density

########## Functions f(U), Q(t), T(U)

def f(U, gamma):
    ''' Returns f(U) for all x, at a certain time. '''
    f_rho = U["m"]
    f_m = (1.5 - gamma / 2) * U["m"]**2 / U["rho"] + (gamma - 1) * U["E"]
    f_E = gamma * U["E"] * U["m"] / U["rho"] + (0.5 - gamma / 2) * U["m"]**3 / (U["rho"]**2)
    return {"rho": f_rho, "m": f_m, "E": f_E}

def Q(t, mode=mode): 
    """ Returns the volumetric energy source density Q(t) (J/(sm^3)) for a given time (s).
        'daylight_sine': sinusoidal during the day only
        'fullday_sine' : sinusoidal across 24-hour cycle (cut to zero at night) """

    T = 86400  # seconds in a day
    t_day = t % T

    if mode == 'no_source':
        return 0.0

    if mode == 'daylight_sine' and day_start <= t_day <= day_end:
        omega = np.pi / (day_end - day_start)
        phase = t_day - day_start
        return max(Q_max * np.sin(omega * phase), 0.0)

    if mode == 'fullday_sine' and day_start <= t_day <= day_end:
        omega = np.pi / T
        return max(Q_max * np.sin(omega * t_day), 0.0)

    return 0.0

def temp(U, R=287):
    """ Return temperature in Celcius (Ideal Gas assumption).
        Not adequate even for real data (with real T, P, rho). """
    # return (0.029 * 9.81 * 42) / (np.log(1.225 / U["rho"]) * 8.314) - 273.15 # for Kaunas
    # e_s = 872 # Pa
    # e = 0.93 * e_s
    # w = 0.622 * e / (U["P"] - e)
    # humid_q = w / (1 + w)                                              
    # R = 287.05 * (1 - humid_q) + 461.5 * humid_q     
    # print("R: ", R)      
    return U["P"] / (U["rho"] * R) - 273.15
    # c_v = 718 # specific heat capacity for air [J/(kg * K)]
    # return (U["E"] - U["m"]**2 / (2 * U["rho"])) / (U["rho"] * c_v) - 273.15


########## Updates

def d_t_new(d_x, gamma, U):
    ''' Returns the maximum time step n+1, according to CFL conditions. '''
    max_vel = np.max(U["m"] /  U["rho"])                    # air velocity
    max_c_n = np.max(np.sqrt(gamma * U["P"] / U["rho"]))    # sound velocity
    return d_x / (max_c_n + max_vel)

def P(gamma, U):
    ''' Returns P (pressure) for all x, at a certain time.  '''
    return (gamma - 1) * (U["E"] - U["m"]**2 / (2 * U["rho"]))

def lf_step(d_x, gamma, U, n, t, d_t, boundary = 'const_gradient'):
    ''' Lax–Friedrichs method.
        Returns the feature values for n+1. '''
    f_U = f(U, gamma)
    if print_steps: print("n inside LF step: ", n)

    U_new = {}
    for key in ("rho", "m", "E"):
        # Pad each variable with second/second-to-last value
        # "Equivalent" to ssetting u(x=0) and u(x=N) for all t
        if real_data == True:
            U_pad = np.concatenate(([U_b[n][key]], U[key], [U_c[n][key]]))
            f_U_pad = np.concatenate(([f(U_b[n], gamma)[key]], f_U[key], [f(U_c[n], gamma)[key]]))
        elif boundary == 'neighbor':
            U_pad = np.concatenate(([U[key][1]], U[key], [U[key][-2]]))
            f_U_pad = np.concatenate(([f_U[key][1]], f_U[key], [f_U[key][-2]]))
        elif boundary == 'const_gradient':
            U_pad = np.concatenate(([2*U[key][0] - U[key][1]], U[key], [2*U[key][-1] - U[key][-2]]))
            f_U_pad = np.concatenate(([2*f_U[key][0] - f_U[key][1]], f_U[key], [2*f_U[key][-1] - f_U[key][-2]]))

        # Apply Lax–Friedrichs update
        # U_pad[:-2][i] = U[i - 1] and U_pad[2:][i]  = U[i + 1] because U_pad[1:-1] is U_new (U[i])
        U_new[key] = (
            0.5 * (U_pad[:-2] + U_pad[2:]) -
            d_t / (2 * d_x) * (f_U_pad[2:] - f_U_pad[:-2])
        )

    # particular solution ("applied" to all x points)
    Q_t = np.full_like(U["E"], Q(t))
    U_new["E"] += d_t * Q_t
    
    U_new["P"] = P(gamma, U_new)
    return U_new

########## Feature Values U

U_0 = {
    "rho": rho_0,         # initial density
    "m": m_0,             # initial momentum
    "E": E_0              # initial total energy
}
U_0["P"] = P(gamma, U_0)  # pressure
U_list = [U_0]            # initial values

vars_dict = {
    key: label for key, label in {
        "rho": ("Density", "kg/m³"),
        "m": ("Momentum density", "kg/(m²·s)"),
        "E": ("Energy density", "J/m³"),
        "P": ("Pressure", "Pa")
    }.items()
}
variables = list(vars_dict.keys())

########## Run

def run_simulation(gamma, d_x, U_list, boundary):
    ''' Features evolution with M the number of time steps.
        Returns U_list (U for all times). '''
    print("\n Running the simulation...")
    t_end = 86400
    t_list = [0.0]
    for n in range(int(M)):
    # while t_list[-1] < t_end:
        U_last = U_list[-1]
        if real_data == True:
            if d_t_fixed < d_t_new(d_x, gamma, U_last):
                d_t = d_t_fixed
                # print("Condition secured: ", d_t_fixed, " < ", d_t_new(d_x, gamma, U_last))
            else:
                print("Condition not secured because: ", d_t_fixed, " < ", d_t_new(d_x, gamma, U_last))
        else:
            d_t = d_t_new(d_x, gamma, U_last)
            print("Condition not secured because: ", d_t_fixed, " < ", d_t_new(d_x, gamma, U_last))
        t_current = t_list[-1] + d_t

        # Run Lax–Friedrichs step
        U_new = lf_step(d_x, gamma, U_last, n, t_current, d_t, boundary)
        U_list.append(U_new)
        t_list.append(t_current)

    return U_list, t_list

U_list, t_list = run_simulation(gamma, d_x, U_list, boundary) # an array (for each t) of dictionaries, where each key is associated with an array (each x)
temp_list = np.array([temp(U, R = R) for U in U_list]) # an array (for each t) of arrays, where each element is T for each x

# Real and Predicted Temperatures
T_K_pred = [float(temp_list[t][x_K]) for t in range(len(temp_list))]

########## Plot Options

def hist(var, data, show=False):
    ''' Plots and saves a histogram for all t and x. '''
    print(f"\n Creating the histogram for {var}...")
    name, unit = vars_dict.get(var, (var, ""))
    plt.figure(figsize=(20, 14))
    plt.hist(data.flatten(), bins=200, color='steelblue', edgecolor='black', log=True)
    plt.xlabel(f'{name} [{unit}]')
    plt.ylabel('Frequency (log scale)')
    plt.title(f'{var} ({name}) Distribution (log scale)')
    plt.tight_layout()
    plt.savefig(f"hist_{var}_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def hist_all(variables=variables, U_list=U_list, show=False):
    '''Plots and saves a figure with the histograms for all variables.'''
    print(f"\n Creating the histograms plot...")
    _, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for ax, var in zip(axes, variables):
        data = np.array([U[var] for U in U_list])
        name, unit = vars_dict.get(var, (var, ""))
        ax.hist(data.flatten(), bins=200, color='steelblue', edgecolor='black', log=True)
        ax.set_xlabel(f'{name} [{unit}]')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_title(f'{var} ({name}) Distribution')

    plt.tight_layout()
    plt.savefig(f"all_hist_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def heat(var, data, show=False):
    ''' Plots and saves a heat map, with x on the x axis and t on the y axis. '''
    print(f"\n Creating the heat map for {var}...")
    name, unit = vars_dict.get(var, (var, ""))
    plt.figure(figsize=(20, 14))
    plt.imshow(
        data,
        extent=[b, c, 0, M],
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=np.min(data),
        vmax=np.max(data),
    )
    plt.colorbar(label=f"{name} [{unit}]")
    plt.xlabel('Space (x)')
    plt.ylabel('Time step (t)')
    plt.title(f'{var} ({name}) Evolution Over Space and Time')
    plt.tight_layout()
    plt.savefig(f"heatmap_{var}_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def heat_all(variables=variables, U_list=U_list, show=False):
    '''Plots and saves a figure with the heatmaps for all variables.'''
    print(f"\n Creating the heatmaps plot...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for ax, var in zip(axes, variables):
        data = np.array([U[var] for U in U_list])
        name, _ = vars_dict.get(var, (var, ""))
        im = ax.imshow(
            data,
            extent=[b, c, 0, M],
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=np.min(data),
            vmax=np.max(data),
        )
        ax.set_xlabel('Space (x)')
        ax.set_yticks(np.linspace(0, M, 25))
        ax.set_yticklabels([f"{int(i * d_t_fixed / 3600)} h" for i in np.linspace(0, M, 25)])
        ax.set_ylabel('Time step (t)')
        ax.set_title(f'{var} ({name}) Evolution')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.plot()
    plt.tight_layout()
    plt.savefig(f"all_heatmap_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def linecharts_loc_all(variables=variables, U_list=U_list, x_K = int(x_K), show=False):
    '''Plots and saves a file with subplots, for all variables, with x the time and y the variable value.'''
    print(f"\n Creating the linecharts (one location) plot...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for ax, var in zip(axes, variables):
        name, unit = vars_dict.get(var, (var, ""))
        ax.plot(range(0, len(U_list)), [U_list[n][var][x_K] for n in range(0, len(U_list))], label=f'simulated {var}')
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{var} ({name}) across Time for Kaunas')

    for ax in axes[-2:]: # axis only for the two bottom ones
        ax.set_xlabel('Time (s)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Kaunas Line Charts', fontsize=16)
    fig.legend(handles, labels, loc='center right', title='Time steps')

    plt.savefig(f"all_linecharts_Kaunas_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def linecharts_loc_compare_all(variables=variables, U_list=U_list, U_K = U_K, x_K = int(x_K), show=False):
    '''Plots and saves a file with subplots, for all variables, with x the time, two variable lines (simulated and real).'''
    print(f"\n Creating the linecharts (one location) comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for ax, var in zip(axes, variables):
        name, unit = vars_dict.get(var, (var, ""))
        ax.plot(range(len(U_list)), [U_list[n][var][x_K] for n in range(len(U_list))],
            label=f'Simulated', linestyle='--', color='tab:blue')
        ax.plot(range(len(U_K)), [U_K[n][var] for n in range(len(U_K))],
                label='Real', linestyle='-', color='tab:orange')
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'Comparison of {var} ({name}) over Time in Kaunas')


    for ax in axes[-2:]: # axis only for the two bottom ones
        ax.set_xlabel('Time (s)')
    
    tick_positions = np.linspace(0, len(U_list) - 1, 8).astype(int)
    tick_labels = [f"{int(i * d_t_fixed / 3600)} h" for i in tick_positions]

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)


    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle('Kaunas Line Charts (Simulated vs. Real Data)', fontsize=16)
    fig.legend(handles, labels, loc='center right', title='Time steps')

    plt.savefig(f"all_linecharts_Kaunas_comp_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def linecharts_loc_compare_temp(T_K_real = T_K_real, T_K_pred = T_K_pred, d_t_fixed = d_t_fixed, mode = mode, boundary = boundary, show=False):
    '''Plots and saves a comparison plot of simulated vs real temperature at location x_K over time.'''
    print("\nCreating temperature comparison plot (one location)...")

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(T_K_pred)), T_K_pred, label='Predicted',
             linestyle='--', color='tab:blue', linewidth=2)
    plt.plot(range(len(T_K_real)), T_K_real, label='Real',
             linestyle='-', color='tab:orange', linewidth=2)     

    plt.xlabel('Time')
    plt.ylabel('Temperature [°C]')
    plt.title(f'Temperature Comparison over Time at Kaunas')
    
    tick_positions = np.linspace(0, len(U_list) - 1, 8).astype(int)
    tick_labels = [f"{int(i * d_t_fixed / 3600)} h" for i in tick_positions]
    plt.xticks(tick_positions, tick_labels)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"T_Kaunas_comp_{mode}_{boundary}.png")
    if show:
        plt.show()
    plt.close()

def linecharts(variables=variables, U_list=U_list, show=False):
    ''' Plots and saves a line chart, with x on the x axis and a line for each t. '''
    print(f"\n Creating the linecharts...")
    for var in variables:
        name, unit = vars_dict.get(var, (var, ""))
        plt.figure(figsize=(10, 6))
        for t in range(min(10, len(U_list))):
            plt.plot(space_x, U_list[t][var], label=f't step {t}')
        plt.xlabel('Space (x)')
        plt.ylabel(f'{name} [{unit}]')
        plt.title(f'{var} ({name}) across Space')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"linecharts_{var}_{mode}_{boundary}.png")
        if show: plt.show()
        plt.close()

def linecharts_all(variables=variables, U_list=U_list, show=False):
    '''Plots and saves a file with subplots, for all variables, with lines for each time step.'''
    print(f"\n Creating the linecharts plot...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for ax, var in zip(axes, variables):
        name, unit = vars_dict.get(var, (var, ""))
        for t in range(min(10, len(U_list))):
            ax.plot(space_x, U_list[t][var], label=f't step {t}')
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{var} ({name}) across Space')

    for ax in axes[-2:]: # axis only for the two bottom ones
        ax.set_xlabel('Space (x)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Time Steps Line Charts Over Space', fontsize=16)
    fig.legend(handles, labels, loc='center right', title='Time steps')

    plt.savefig(f"all_linecharts_{mode}_{boundary}.png")
    if show: plt.show()
    plt.close()

def animations(variables=variables, U_list=U_list, show=False):
    ''' Plots and saves an animation for each variable. '''
    print(f"\n Creating the animations for each variable...")
    for var in variables:
        name, unit = vars_dict.get(var, (var, ""))
        fig, ax = plt.subplots(figsize=(20, 14))

        var_line, = ax.plot(space_x, U_list[0][var], label=f'{var}')
        temp_line, = ax.plot(space_x, temp_list[0], label='Temperature (°C)', color='red', linestyle='--')
        
        ax.set_xlim(space_x.min(), space_x.max())
        ax.set_ylim(
            min(np.min(U[var]) for U in U_list + [temp_list]),
            max(np.max(U[var]) for U in U_list + [temp_list])
        )
        ax.set_xlabel('Space (x)')
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{var} ({name}) across Space over Time')
        ax.legend(loc='upper right')

        def update(frame):
            var_line.set_ydata(U_list[frame][var])
            temp_line.set_ydata(temp_list[frame])
            ax.set_title(f'{var} and Temperature Evolution across Space — Time {(t_list[frame]/3600):.2f} h')
            return var_line, temp_line
    
        ani = FuncAnimation(fig, update, frames=len(U_list), interval=300, blit=True)
        ani.save(f"animation_{var}.gif", writer=PillowWriter(fps=120))
        if show: plt.show()
        plt.close()

def animation_all(variables=variables, space_x=space_x, U_list=U_list, show=False):
    ''' Plots and saves a single animation with subplots for all variables. '''
    print(f"\n Creating the animation with all variables...\n")
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 16), sharex=True)
    
    lines = []

    for ax, var in zip(axes, variables):
        name, unit = vars_dict.get(var, (var, ""))
        line, = ax.plot(space_x, U_list[0][var], label=f'{var} ({name})')
        ax.set_ylabel(f'{var} [{unit}]')
        ax.set_ylim(
            min(np.min(U[var]) for U in U_list),
            max(np.max(U[var]) for U in U_list)
        )
        ax.legend(loc='upper right')

        lines.append(line)

    axes[-1].set_xlabel('Space (x)')
    suptitle = fig.suptitle(f"Evolution over Space — Time step 0", fontsize=16)

    def update(frame):
        print(f"frame {frame}")
        for line, var in zip(lines, variables):
            line.set_ydata(U_list[frame][var])
        suptitle.set_text(f"Evolution over Space — Time {t_list[frame]:.2f} s")
        return lines + [suptitle]

    ani = FuncAnimation(fig, update, frames=len(U_list), interval=300, blit=False)
    ani.save(f"all_animation_{mode}_{boundary}.gif", writer=PillowWriter(fps=5))
    if show: plt.show()
    plt.close()

def animation_all_temp(variables=variables, space_x=space_x, U_list=U_list, U_K = U_K, show=False):
    ''' Plots and saves a single animation with subplots for all variables plus temperature as the last subplot. '''
    print(f"\n Creating the animation with all variables plus temperature...")

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    lines = []
    dots_real = []

    kaunas_x = space_x[x_K]  # x-position of Kaunas

    for ax, var in zip(axes[:-1], variables):
        name, unit = vars_dict.get(var, (var, ""))
        line, = ax.plot(space_x, U_list[0][var], # check
                        label=f'Simulated', linestyle='--', color='tab:blue')
        dot_real = ax.scatter(kaunas_x, U_K[0][var],
                       label='Real (Kaunas)', color='tab:orange', marker='o')
        ax.set_ylabel(f'{var} [{unit}]')
        kaunas_values = [U_K[t][var] for t in range(len(U_K))]  # real data at Kaunas over time
        ax.set_ylim(
            min(min(np.min(U[var]) for U in U_list), min(kaunas_values)),
            max(max(np.max(U[var]) for U in U_list), max(kaunas_values))
        )
        ax.legend(loc='upper right')
        lines.append(line)
        dots_real.append(dot_real)
    
    temp_ax = axes[-1]
    temp_line, = temp_ax.plot(space_x, temp_list[0], label='Simulated Temperature (°C)', linestyle='--', color='red') # can change
    temp_dot_real = temp_ax.scatter(kaunas_x, T_K_real[0],
                                label='Real Temperature (°C)', color='tab:orange', marker='o')
    temp_ax.set_ylabel('Temperature [°C]')
    temp_ax.set_ylim(min(np.min(temp_list), np.min(T_K_real)), max(np.max(temp_list), np.max(T_K_real)))
    temp_ax.legend(loc='upper right')

    axes[-1].set_xlabel('Space (x)')
    suptitle = fig.suptitle(f"Evolution over Space — Time step 0 ({mode})", fontsize=16)

    def update(frame):
        print(f"frame {frame}")
        for line, dot_real, var in zip(lines, dots_real, variables):
            line.set_ydata(U_list[frame][var])
            dot_real.set_offsets([[kaunas_x, U_K[frame][var]]])
        temp_line.set_ydata(temp_list[frame])
        temp_dot_real.set_offsets([[kaunas_x, T_K_real[frame]]])
        total_seconds = t_list[frame]
        # time in h minutes
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        suptitle.set_text(f"Evolution over Space (Simulated vs. Real Data) — Time {hours}h {minutes}m ({mode})")
        return lines + dots_real + [temp_line, temp_dot_real, suptitle]
    
    # instead of len(U_list), using list(range(0, len(U_list), 60*5)) should have around 24 * 20 = 480 frames
    ani = FuncAnimation(fig, update, frames=list(range(0, len(U_list), 60*5)), interval=300, blit=False)
    print("\n Saving the comparison animation... \n")
    ani.save(f"all_animation_temp_comp_{mode}_{boundary}.gif", writer=PillowWriter(fps=5))
    if show: plt.show()
    plt.close()

########## Plot Source Term

def source(show=False):
    nr_days = 1
    t_space_s = np.linspace(0, 86400*nr_days, 1000)
    t_space_h = t_space_s / 3600
    for mode in modes:
        Q_values = [Q(t, mode = mode) for t in t_space_s]

        plt.figure(figsize=(20, 14))
        plt.plot(t_space_h, Q_values, label='Q(t) Source Term')
        plt.xlabel('Time of Day (hours)')
        plt.ylabel('Source Term Q(t)')
        plt.title(f'Energy Source Term Over One Time ({mode} mode, boundary {boundary})')
        plt.grid(True)
        plt.xlim(0, 24*nr_days)
        plt.tight_layout()
        plt.savefig(f"source_{mode}_{boundary}.png")
        if show: plt.show()
        plt.close()

########## Run and Visualize

# var_plot = "P"
# data_plot = np.array([U[var_plot] for U in U_list])
# hist(var = var_plot, data = data_plot)
# heat(var = var_plot, data = data_plot)
# linecharts()
# animations()
# animation_all(show = True)
# linecharts_loc_all()

# source()
# print(t_list)

hist_all()
heat_all()
linecharts_loc_compare_all(show = True)
linecharts_loc_compare_temp(show = True)
linecharts_all()
animation_all_temp(show = True)

# Real and Predicted Temperatures
# print("T_K_real[:10]: ", T_K_real)
# print("T_K_pred[:10]: ", T_K_pred)

with open("T_K_pred.txt", "w") as f:
    f.write("T_K_pred: " + str(T_K_pred))