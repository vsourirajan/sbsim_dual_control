import papermill as pm
import os 
import resource

# Set soft memory limit to 32GB
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (32 * 1024 * 1024 * 1024, hard))

base_dir = "/burg/home/ssa2206/sbsim_dual_control/smart_control/notebooks/"
pathify = lambda fpath : os.path.join(base_dir,fpath)
pm.execute_notebook(
    input_path= pathify('SAC_model_assisted_rollout.ipynb'), 
    output_path=pathify('SAC_Demo_res.ipynb'), 
    kernel_name='large_memory_kernel',
    engine='subprocess'  # Can help with memory management
    #parameters=dict(alpha=1, beta=2)
)
