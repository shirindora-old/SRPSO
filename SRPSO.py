import scipy as sp

def SRPSO(data, var_info, obj_func, pso_params, user_best):
    # Read the data
    tr_dat = data['tr_dat']
    tr_cls = data['tr_cls']
    ts_dat = data['ts_dat']
    ts_cls = data['ts_cls']

    # Setup PSO parameters
    swarm_size = pso_params[0].astype(int)  # Number of particles in an iteration
    max_IC = pso_params[1].astype(int)      # Maximum number of iterations allowed
    IC = 0                                  # Count of iterations completed
    c1 = 1.49445
    c2 = 1.49445
    
    # Information regarding the variables to be optimized
    optimize_var_idx = sp.nonzero(var_info[:, 0] != 2)[0]               # Index of variables to be optimized
    var_count = optimize_var_idx.size                                   # Number of variables to be optimized
    int_var_idx = sp.zeros(var_count, dtype=int)
    const_params = var_info[var_info[:, 0] == 2, 1]                     # Value for variables not to be optimized
    l_bound = sp.tile(var_info[optimize_var_idx, 1], (swarm_size, 1))
    u_bound = sp.tile(var_info[optimize_var_idx, 2], (swarm_size, 1))
    
    # Initialize swarms
    swarm = sp.zeros((swarm_size, var_count))
    for i in range(optimize_var_idx.size):
        current_var = optimize_var_idx[i]
        if var_info[current_var, 0] == 0:    # For real valued variables
            swarm[:, i] = var_info[current_var, 1] + (var_info[current_var, 2] -
                                                      var_info[current_var, 1]) * sp.random.rand(swarm_size);
        elif var_info[current_var, 0] == 1:  # For integer valued variables
            swarm[:, i] = sp.random.randint(var_info[current_var, 1], var_info[current_var, 2], swarm_size)
            int_var_idx[i] = 1

    int_var_idx = int_var_idx == 1
    history = sp.zeros((max_IC, var_count + 1))
    swarm[-1, :] = user_best
    
    # Initialize velocity
    vel = sp.zeros((swarm_size, var_count))
    max_vel = (var_info[optimize_var_idx, 2] - var_info[optimize_var_idx, 1]) * 0.100625
    max_vel = sp.tile(max_vel, (swarm_size, 1))
    
    # Initialize weight. Weight will vary linearly for w_vary_for iterations.
    w = sp.tile(pso_params[2], (swarm_size, var_count))
    w_end = pso_params[3]
    w_vary_for = sp.floor(pso_params[4] * max_IC)
    linear_dec = (pso_params[2] - w_end)/w_vary_for
    
    # Evaluate fitness for each particle
    fitness = sp.zeros(swarm_size)
    for i in range(swarm_size):
        params = sp.concatenate((const_params, swarm[i, :]), axis=1)
        fitness[i] = obj_func(tr_dat, tr_cls, ts_dat, ts_cls, params)

    g_best_ind = sp.argmax(fitness)
    g_best_fitness = fitness[g_best_ind]
    g_best = swarm[g_best_ind, :]
    p_best = swarm
    p_best_fitness = fitness
    current_g_best_idx = g_best_ind
    history[IC, 0:-1] = g_best
    history[IC, -1] = g_best_fitness
    swarm_idx = sp.arange(swarm_size)
    
    while IC < max_IC:
        rand_num_1 = sp.random.rand(swarm_size, var_count)
        rand_num_2 = sp.random.rand(swarm_size, var_count)
        
        non_best_idx = sp.setdiff1d(swarm_idx, current_g_best_idx)
        
        if IC <= w_vary_for:
            w[current_g_best_idx, :] = w[current_g_best_idx, :] + linear_dec
            w[non_best_idx, :] = w[non_best_idx, :] - linear_dec
        
        vel_update_flag = sp.random.rand(swarm_size - 1, var_count) > 0.5
        
        vel[current_g_best_idx, :] = w[current_g_best_idx, :] * vel[current_g_best_idx, :]
        vel[non_best_idx, :] = w[non_best_idx, :] * vel[non_best_idx, :] + \
                c1 * (rand_num_1[non_best_idx, :] * (p_best[non_best_idx, :] - swarm[non_best_idx, :])) + \
                c2 * (rand_num_2[non_best_idx, :] * vel_update_flag *
                (sp.tile(g_best, (swarm_size - 1, 1)) - swarm[non_best_idx, :]))
          
        vel = sp.minimum(max_vel, sp.maximum(-max_vel, vel))
        swarm = swarm + vel
        swarm[:, int_var_idx] = sp.around(swarm[:, int_var_idx])
        
        swarm = sp.minimum(u_bound, sp.maximum(l_bound, swarm))
        
        for i in range(swarm_size):
            params = sp.concatenate((const_params, swarm[i, :]), axis=1)
            fitness[i] = obj_func(tr_dat, tr_cls, ts_dat, ts_cls, params)
        
        update_p_best = fitness > p_best_fitness
        p_best[update_p_best, :] = swarm[update_p_best, :]
        p_best_fitness[update_p_best] = fitness[update_p_best]
        
        current_g_best_idx = sp.argmax(fitness)
        current_g_best_fitness = fitness[current_g_best_idx]
        if current_g_best_fitness > g_best_fitness:
            g_best_fitness = current_g_best_fitness
            g_best = swarm[current_g_best_idx, :]

        history[IC, 0:-1] = g_best
        history[IC, -1] = g_best_fitness

        print('Iteration: ' + str(IC) + 'Best fitness ' + str(g_best_fitness))
        print('Params: ' + str(g_best) + '\n\n')
        IC = IC + 1