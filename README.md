# Energy ABM

The Energy Agent-Based Model has been developed to explore the implications of assumptions on informal institutions in the adoption of energy-efficient technologies by households.
The model focuses specifically on changes in estimated greenhouse gas emissions (GHG) and investment related to PV panel installation by households in the Netherlands.
Results generated for this purpose are desribed in (Davidson et al., 2024) [[1]](#1).

The model contains three types of household decision-making modules: rational, behavioral and behavioral with social influence. In the rational decision-making, agents only consider the financial aspects of installing PV panels, whereas the behavioral decision-making process follows the Theory of Planned Behavior (TPB) in which agents based their decisions also on their attitude towards the behavior and on subjective norms, which refer to the opinions and actions of others. In addition, it is possible to add opinion dynamics to the model, in which social influence affects the previously mentioned attitudes of the households.

The model can be run from the `run.py` file, in which the user can specify the number of runs and agents, as well as the type of decision-making and whether to include opinion dynamics or not.

## Documentation
A full description of the model can be found in the attached `ODD_Description_Energy_Model.pdf` file.

## References
<a id="1">[1]</a>
M.R. Davidson, T. Filatova, W. Peng, L. Verbeek & F. Kucuksayacigil (2024). [Simulating institutional heterogeneity in sustainability science, Proc. Natl. Acad. Sci. U.S.A. 121 (8) e2215674121](https://doi.org/10.1073/pnas.2215674121).
