# 02456_l2rpn

To run the training, set the configuration in config.json and run train.py with arguments -c config.json and -n NAME. For example: python train.py -c config.json -n my_awesome_dddqn

To test the trained agents, run test.py with argument -c config.json and -n NAME. Where NAME should be the name of one of the trained agents in trained_models/ excluding the suffix '_policy_net_last.pth'. For example: python test.py -c config.json -n ddqn_500

To run an interactive version of the results, use grid2viz with argument --agents_path ./runner_agents

# Dependencies

- grid2op (https://github.com/rte-france/Grid2Op)
- lightsim2grid (https://github.com/BDonnot/lightsim2grid)
- grid2viz (https://github.com/mjothy/grid2viz)

If avoiding a local run and using a colab session, these dependencies can be installed easily with the commands:

```
!git clone https://github.com/jmontalvo94/02456_l2rpn.git
!pip install grid2op
!git clone https://github.com/BDonnot/lightsim2grid.git
!cd lightsim2grid ; git submodule init ; git submodule update ; make ; pip install -U pybind11 ; pip install -U .
```

and then running, for example:

```
!cd 02456_l2rpn ; python test.py -c config.json -n ddqn_500
```
