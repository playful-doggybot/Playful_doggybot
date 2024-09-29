import subprocess
import importlib

def run_experiment():
    subprocess.run(["python", "legged_gym/scripts/train.py", "--task", "go2_remote", "--headless"])

def modify_config_variable(new_value, variable_name):
    # 修改config.py中的变量值
    with open("legged_gym/envs/go2/go2_remote_config.py", "r") as f:
        lines = f.readlines()

    with open("legged_gym/envs/go2/go2_remote_config.py", "w") as f:
        for line in lines:
            variable = None
            for name, idx in variable_name.items():
                if line.strip().startswith(name):
                    variable = name
            if variable:
                f.write('    ' * 3 + variable + ' = ' + repr(new_value[variable_name[variable]]) + '\n')
            else:
                f.write(line)


names = {"lin_vel_l2norm": 0, "legs_energy_substeps": 1}
for val in [[0.5,-6e-07], [0.75, -6e-07], [1.,-6e-07], [0.25, -6e-07]]:
    modify_config_variable(val, names)
    run_experiment()