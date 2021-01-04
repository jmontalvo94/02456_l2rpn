# 02456_l2rpn

To run the training, set the configuration in config.json and run train.py with arguments -c config.json and -n NAME.

To test the trained agents, run test.py with argument -c config.json and -n NAME. Where NAME should be the name of one of the trained agents in trained_models/

To run an interactive version of the results, use grid2viz with argument --agents_path ./runner_agents

# Dependencies

grid2op (https://github.com/rte-france/Grid2Op)
lightsim2grid (https://github.com/BDonnot/lightsim2grid)
grid2viz (https://github.com/mjothy/grid2viz)
