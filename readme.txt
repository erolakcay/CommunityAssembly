Python code for simulations from "The balance of interaction types determines the assembly and stability of ecological communities" by Jimmy Qian and Erol Akçay, 2019.

The code runs simulations of our model of community assembly. Data for communities are saved as Numpy .npz zipped archives of files named after the variables they contain.

"community-assembly-stability.py" simulates a single community whose parameters are defined within the script and by a command line argument. The script should be run with a command line argument, an integer between 0 to 593, that determines the parameter values that are used to simulate the community. As described in the Methods, there are a total of 66*3*3=594 combinations we consider. There are 66 combinations of interaction types (proportions of mutualism, competition, and exploitation). There are 3 levels of connectivity and 3 values of the half-saturation constant. Data from these simulations were used to create all figures.

The additional options that can be passed into the simulation are described in the script. These options can be chosen by initializing the Community object with desired parameters in the function run_simulation. For example, the default is to run the UIM but the IIM is implemented by setting pool_interactions = True. The intrinsic growth rate is kept constant for all species, unless variable_r = True in which case the intrinsic growth rate can vary between -1 and 1 based on the proportion of predatory interactions (this data is not shown in the manuscript).


