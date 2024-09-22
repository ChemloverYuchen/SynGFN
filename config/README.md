# Hydra configuration files

This directory contains `yaml` files used to parameterize `gflownet` trainings using Hydra.

The main.yaml file provides a set of basic adjustable parameters. For different modules, such as `env` and `policy`, users can further include other configurable parameters in main.yaml for unified adjustments.

# Important Note
Please note that when you need to change the block library scale, don't forget to update the file names for `smiles_list` and `mask_dict` in the gflownet/envs/building_block.py.