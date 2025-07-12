import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

modes = ['no_source', 'daylight_sine', 'fullday_sine']        # Q(t) type of function
boundaries = ['neighbor', 'const_gradient', 'distribution']   # boundary conditions type on x

mode = modes[0]
boundary = boundaries[0]

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

gamma = 1.4                                                  # heat capacity (dry air at room temperature)
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
    """ Return temperature in Celcius (Ideal Gas assumption)."""
    return U["P"] / (U["rho"] * R) - 273.15

########## Updates

def d_t_new(d_x, gamma, U):
    ''' Returns the time step n+1. '''
    max_c_n = np.max(np.sqrt(gamma * U["P"] / U["rho"]))
    return d_x / max_c_n

def P(gamma, U):
    ''' Returns P (pressure) for all x, at a certain time.  '''
    return (gamma - 1) * (U["E"] - U["m"]**2 / (2 * U["rho"]))

def lf_step(d_x, gamma, U, t, d_t, boundary = 'const_gradient'):
    ''' Lax–Friedrichs method.
        Returns the feature values for n+1. '''
    f_U = f(U, gamma)

    U_new = {}
    for key in ("rho", "m", "E"):
        # Pad each variable with second/second-to-last value
        # "Equivalent" to ssetting u(x=0) and u(x=N) for all t
        if boundary == 'neighbor':
            U_pad = np.concatenate(([U[key][1]], U[key], [U[key][-2]]))
            f_U_pad = np.concatenate(([f_U[key][1]], f_U[key], [f_U[key][-2]]))
        if boundary == 'const_gradient':
            U_pad = np.concatenate(([2*U[key][0] - U[key][1]], U[key], [2* U[key][-1] - U[key][-2]]))
            f_U_pad = np.concatenate(([2*f_U[key][0] - f_U[key][1]], f_U[key], [2*f_U[key][-1] - f_U[key][-2]]))

        # Apply Lax–Friedrichs update
        # U_pad[:-2][i] = U[i - 1] and U_pad[2:][i]  = U[i + 1]
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
    for _ in range(M):
    # while t_list[-1] < t_end:
        U_last = U_list[-1]
        d_t = d_t_new(d_x, gamma, U_last)
        t_current = t_list[-1] + d_t

        # Run Lax–Friedrichs step
        U_new = lf_step(d_x, gamma, U_last, t_current, d_t, boundary)
        U_list.append(U_new)
        t_list.append(t_current)

    return U_list, t_list

U_list, t_list = run_simulation(gamma, d_x, U_list, boundary)
temp_list = np.array([temp(U) for U in U_list])

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
        ax.set_ylabel('Time step (t)')
        ax.set_title(f'{var} ({name}) Evolution')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"all_heatmap_{mode}_{boundary}.png")
    if show: plt.show()
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
            ax.set_title(f'{var} and Temperature Evolution across Space — Time {t_list[frame]:.2f} s')
            return var_line, temp_line
    
        ani = FuncAnimation(fig, update, frames=len(U_list), interval=300, blit=True)
        ani.save(f"animation_{var}.gif", writer=PillowWriter(fps=30))
        if show: plt.show()
        plt.close()

# def animation_all_temp(variables=variables, space_x=space_x, U_list=U_list, show=False):
#     ''' Plots and saves a single animation with subplots for all variables. '''
#     print(f"\n Creating the animation with all variables...\n")
#     fig, axes = plt.subplots(len(variables), 1, figsize=(10, 16), sharex=True)
    
#     lines = []
#     temp_lines = []

#     for ax, var in zip(axes, variables):
#         name, unit = vars_dict.get(var, (var, ""))
#         line, = ax.plot(space_x, U_list[0][var], label=f'{var} ({name})')
#         temp_line, = ax.plot(space_x, temp_list[0], label='Temperature (°C)', color='red', linestyle='--')
#         ax.set_ylabel(f'{var} [{unit}]')
#         ax.set_ylim(
#             min(min(np.min(U[var]) for U in U_list), np.min(temp_list)),
#             max(max(np.max(U[var]) for U in U_list), np.max(temp_list))
#         )
#         ax.legend(loc='upper right')

#         lines.append(line)
#         temp_lines.append(temp_line)

#     axes[-1].set_xlabel('Space (x)')
#     suptitle = fig.suptitle(f"Evolution over Space — Time step 0 ({mode})", fontsize=16)

#     def update(frame):
#         print(f"frame {frame}")
#         for line, var in zip(lines, variables):
#             line.set_ydata(U_list[frame][var])
#         for temp_line in temp_lines:
#             temp_line.set_ydata(temp_list[frame])
#         suptitle.set_text(f"Evolution over Space — Time {t_list[frame]:.2f} s ({mode})")
#         return lines + temp_lines + [suptitle]

#     ani = FuncAnimation(fig, update, frames=len(U_list), interval=300, blit=False)
#     ani.save(f"all_animation_temp_{mode}_{boundary}.gif", writer=PillowWriter(fps=5))
#     if show: plt.show()
#     plt.close()

def animation_all_temp(variables=variables, space_x=space_x, U_list=U_list, show=False):
    ''' Plots and saves a single animation with subplots for all variables plus temperature as the last subplot. '''
    print(f"\n Creating the animation with all variables plus temperature...\n")

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    lines = []

    for ax, var in zip(axes[:-1], variables):
        name, unit = vars_dict.get(var, (var, ""))
        line, = ax.plot(space_x, U_list[0][var], label=f'{var} ({name})')
        ax.set_ylabel(f'{var} [{unit}]')
        ax.set_ylim(
            min(np.min(U[var]) for U in U_list),
            max(np.max(U[var]) for U in U_list)
        )
        ax.legend(loc='upper right')
        lines.append(line)
    
    temp_ax = axes[-1]
    temp_line, = temp_ax.plot(space_x, temp_list[0], label='Temperature (°C)', color='red')
    temp_ax.set_ylabel('Temperature [°C]')
    temp_ax.set_ylim(np.min(temp_list), np.max(temp_list))
    temp_ax.legend(loc='upper right')

    axes[-1].set_xlabel('Space (x)')
    suptitle = fig.suptitle(f"Evolution over Space — Time step 0 ({mode})", fontsize=16)

    def update(frame):
        print(f"frame {frame}")
        for line, var in zip(lines, variables):
            line.set_ydata(U_list[frame][var])
        temp_line.set_ydata(temp_list[frame])
        suptitle.set_text(f"Evolution over Space — Time {t_list[frame]:.2f} s ({mode})")
        return lines + [temp_line, suptitle]

    ani = FuncAnimation(fig, update, frames=len(U_list), interval=300, blit=False)
    ani.save(f"all_animation_temp_{mode}_{boundary}.gif", writer=PillowWriter(fps=5))
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


########## Plot Source Term

def source(show=False):
    nr_days = 5
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

# source()
# print(t_list)

hist_all()
heat_all()
linecharts_all()
animation_all_temp(show = True)