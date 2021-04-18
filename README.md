# Affine-Wealth-Model-2-Python
A Monte Carlo implementation of the Affine Wealth Model, as seen in "The Inescapable Casino" research paper. Additional parameters were added for simulating the effects of segregation and skill advantage. An accompanying paper is also included, showing results of these new parameters.

## Requirements
Tested using Python 3.9.1
Additional libraries required:
  -numpy
  -matplotlib
  -scipy
  
## Usage
Run the Affine_Wealth_Model_2.py python file. It will save images of graphs showing the results of the simulation inside a folder 'graphs' in the same directory as where Affine_Wealth_Model_2.py is. Each subfolder in 'graphs' is named based off of the parameters input to the given simulation, with subfolders inside each of those subfolders named based off of the time the simulation was run. The command prompt will display progress of the simulation. Convergence may occur slower or faster depending on the parameters input to the model. It is best to use very small values for chi and zeta, to ensure fast convergence. If chi is set to high, the simulation may fail and restart with a smaller value of delta t. Convergence is determined by the Gini coefficient (and Steve coefficient if tau is not 0) decreasing across consecutive check points. A check point is given by the 'step_count' variable, and determines how often the simulation will calculate properities of the wealth distribution and create acompanying graphs showing those properties.

## Terms of Use
MIT license. Appropriate credit given is appreciated, but no necessary! :)
