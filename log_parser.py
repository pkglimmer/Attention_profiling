# THIS SCRIPT IS DEPRECATED! -PEIKUN

import os 
# profile_log_path = '/home/yuxin/code/taylor_logs/W4_N2048_D64_H4_B16.log'
profile_log_path = '/home/yuxin/code/taylor_logs/W1_N2048_D64_H4_B16.log'
COL_WIDTH = 14 # fixed for profiler output log table
suffix = {'ms':1e-3, 'us':1e-6, 's':1}
N_STEPS = 7
step_line_numbers_multiple_worker = [4, 11, 33, 34, 36, 39, 41]
comm_line_numbers = [30, 34, 43]
step_line_numbers_single_worker = [11, 14, 17, 18, 20, 23, 25]
step_time = [0] * N_STEPS
step_time_percs = [0] * N_STEPS

def calc_CUDA_time(lines, start, end, skip):
    '''
    calculate CUDA time and the corresponding percentage of a given interval in the log file
    '''
    time, pct = 0, 0
    for i in range(start, end): 
        if i in skip: continue
        log_str = lines[i][:-3]
        pct += float(log_str[(-4*COL_WIDTH):(-3*COL_WIDTH)][:-1])
        time_str = log_str[(-3*COL_WIDTH):(-2*COL_WIDTH)]
        for s in suffix.keys():
            if time_str.endswith(s):
                time += float(time_str[:-len(s)]) * suffix[s]
                break
    return time, pct

if __name__ == '__main__':
    base_path = '/home/yuxin/code/taylor_logs'
    logs = sorted(os.listdir('/home/yuxin/code/taylor_logs'))
    verbose = True
    for log in logs:
        profile_log_path = os.path.join(base_path, log)
        print(f"Config: {profile_log_path.split('/')[-1].split('.')[0]}")
        with open(profile_log_path) as f:
            lines = [line for line in f]
            parallel = not profile_log_path.split('/')[-1].startswith('W1')
            step_line_numbers = step_line_numbers_multiple_worker if parallel else step_line_numbers_single_worker
            skip = comm_line_numbers if parallel else []
            print('Computation cost:')
            for i in range(N_STEPS):
                start = step_line_numbers[i]
                end = step_line_numbers[i+1] if i<N_STEPS-1 else len(lines)-4
                step_time, step_time_pct = calc_CUDA_time(lines, start, end, skip)
                print(f'step {i} time: {step_time:.3f}s \t step time percentage: {step_time_pct:.2f}%')
                if verbose:
                    print(f'{step_time:.3f} ({step_time_pct:.2f}%)')
                
            if parallel:
                reduce_time,reduce_pct, gather_time, gather_pct = 0, 0, 0, 0
                for k, i in enumerate(comm_line_numbers):
                    step_time, step_time_pct = calc_CUDA_time(lines, i, i+1, skip=[])
                    if k == len(comm_line_numbers)-1:
                        gather_time += step_time
                        gather_pct += step_time_pct
                    else:
                        reduce_time += step_time
                        reduce_pct += step_time_pct
                print('Communication cost:')
                print(f'Reduce time: {reduce_time:.3f}s \t reduce time percentage: {reduce_pct:.2f}%')
                print(f'Gather time: {gather_time:.3f}s \t reduce time percentage: {gather_pct:.2f}%')

            CPU_time = float(lines[-2].split(' ')[-1][:-2])
            CUDA_time = float(lines[-3].split(' ')[-1][:-2])
            total_time = CPU_time + CUDA_time
            print(f'Total time: {total_time:.3f}, CPU time: {CPU_time:.3f}, CUDA time: {CUDA_time:.3f}\n\n')
    
    
                
    
        