import os
import pickle 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

def hamiltonian_fn(coords):
    q1, q2, p1, p2 = np.split(coords,4)
    H = ((p1**2 + 2*p2**2 -2*p1*p2*np.cos(q1 - q2))/(2*(1 + np.sin(q1-q2)**2))) - 2*10*np.cos(q1) - 10*np.cos(q2)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    Dq1, Dq2, Dp1, Dp2 = np.split(dcoords,4)
    S = np.concatenate([Dp1, Dp2, -Dq1, -Dq2], axis=-1)
    return S

def get_trajectory(t_span=[0,3], numberofpouints=100, y0=None, noise_std=0.01, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], num=numberofpouints)
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(4)*np.pi*2

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q1, q2, p1, p2 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    Dq1, Dq2, Dp1, Dp2 = np.split(dydt,4)
    
    # add noise
    q1 += np.random.randn(*q1.shape)*noise_std
    q2 += np.random.randn(*q2.shape)*noise_std
    p1 += np.random.randn(*p1.shape)*noise_std
    p2 += np.random.randn(*p2.shape)*noise_std
    return q1, q2, p1, p2, Dq1, Dq2, Dp1, Dp2, t_eval

def get_dataset_with_cache(**kwargs):
    if not os.path.isfile(f'{THIS_DIR}/data_double_pend.pkl'):
        data = get_dataset(**kwargs)
        with open(f'{THIS_DIR}/data_double_pend.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data
    else:
          with open(f'{THIS_DIR}/data_double_pend.pkl', 'rb') as f:
              data = pickle.load(f)
              return data


def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        if s%10 == 0:
            print(f'Sample number: {s}')
        q1, q2, p1, p2, Dq1, Dq2, Dp1, Dp2, t = get_trajectory(**kwargs)
        xs.append( np.stack( [q1, q2, p1, p2]).T )
        dxs.append( np.stack( [Dq1, Dq2, Dp1, Dp2]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

if __name__ == "__main__":
    t_span=[0, 3]
    numberofpouints=100
    q1, q2, p1, p2, Dq1, Dq2, Dp1, Dp2, t_eval = get_trajectory(t_span=t_span, numberofpouints=numberofpouints, noise_std=0.01)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    x1 = np.sin(q1)
    y1 = -np.cos(q1)

    x2 = np.sin(q2) + x1
    y2 = -np.cos(q2) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i*1*(t_span[1]-t_span[0])/numberofpouints))
        return line, trace, time_text


    ani = animation.FuncAnimation(
        fig, animate, len(y1), interval=1000*(t_span[1]-t_span[0])/numberofpouints, blit=True)
    plt.show() 