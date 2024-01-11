# -*- coding: utf-8 -*-

"""

@author: Liz Verbeek

This file contains a class for Household agents in the EnergyModel.
The class is based on the MESA Agent class, and contains functions for
initialization, decision-making (rational and behavioral) and PV installation.

"""

import numpy as np

from mesa import Agent


class Household(Agent):
    """Household class representing a household agent in the energy model."""

    def __init__(self, model, TPB_weights=None):
        """Initialization of the energy model.

        Args:
            model           : Model object that contains the agent
            TPB_weights     : Dict {"Energy saving method": [TPB weights]}
        """

        super().__init__(model.next_id(), model)

        # -- GENERAL ATTRIBUTES -- #
        self.income = self.model.rng.normal(47052, 25232)
        while self.income < 3264:  # Lower bound = 3264 (unemployment subsidy)
            self.income = self.model.rng.normal(47052, 25232)
        self.savings = 0
        
        # -- DECISION MAKING ATTRIBUTES -- #
        if self.model.decision_making_model == "TPB":
            # Initialize weights for TPB based on population shares
            self.TPB_weights = TPB_weights
            # Initialize TPB attribute values as vector [Att, SN, PBC]
            attitude = self.model.rng_TPB.normal(0.2004, 0.4580)
            self.attitude = {"PV": attitude, "no_PV": -attitude}
            self.social_norm = {"PV": 0, "no_PV": 0}
            self.PBC = {"PV": 0, "no_PV": 0}

        # -- ENERGY-RELATED ATTRIBUTES -- #
        self.energy_use = self.model.rng.normal(2770, 1553)
        while self.energy_use < 0:  # Lower bound = 0 on energy use
            self.energy_use = self.model.rng.normal(2770, 1553)

        self.number_of_PVs = np.ceil(self.energy_use/(0.9 * 370))
        self.PV_costs = 1484 + 428 * self.number_of_PVs
        self.NPV = self.estimate_NPV()

        self.PV_installed = False
        self.CO2_saved = self.CO2_produced = 0

        # -- SOCIAL NETWORK ATTRIBUTES -- #
        self.neighbors = []

    def estimate_NPV(self):
        """Get Net Present Value (NPV) of PV installation.
           Used as decision-making module for rational agents.
        """

        # Get yearly benefits, assuming fixed energy use and price
        benefits_PV = (self.model.energy_cost[0] +
                       self.model.energy_cost[1] * self.energy_use)
        # Get yearly costs (only in first year when investment is made)
        costs_PV = np.zeros(self.model.PV_lifespan)
        costs_PV[0] = self.PV_costs

        NPV_PV = sum((benefits_PV - costs_PV) / self.model.discount_rates)
        NPV_no_PV = sum((costs_PV - benefits_PV) / self.model.discount_rates)

        return NPV_PV, NPV_no_PV

    def TPB(self):
        """Household decision-making module: Theory of Planned Behavior. """

        # Update subjective norm (fraction of neighbors with PV installed)
        self.social_norm["PV"] = (sum(hh.PV_installed for hh in self.neighbors)
                                  / len(self.neighbors))
        self.social_norm["no_PV"] = (sum(not hh.PV_installed for hh in self.neighbors)
                                     / len(self.neighbors))

        NPV = self.NPV / self.model.max_NPV
        self.PBC["PV"], self.PBC["no_PV"] = self.NPV / self.model.max_NPV

        U_PV = np.dot(self.TPB_weights, [self.attitude["PV"],
                                         self.social_norm["PV"],
                                         self.PBC["PV"]])
        U_no_PV = np.dot(self.TPB_weights, [self.attitude["no_PV"],
                                            self.social_norm["no_PV"],
                                            self.PBC["no_PV"]])
        return U_PV, U_no_PV

    def step(self):
        """Defines single Household timestep. """

        # Add percentage of income to yearly savings
        self.savings += self.income * self.model.hh_savings_ratio

        # Keep track of total CO2 saved
        if self.PV_installed:
            self.CO2_saved = self.energy_use * self.model.CO2_emission
            self.CO2_produced = 0
        else:
            self.CO2_produced = self.energy_use * self.model.CO2_emission
            self.CO2_saved = 0

        # Opinion dynamics: update household attitude
        if (self.model.decision_making_model == "TPB"
                and self.model.opinion_dynamics):
            # Query attitudes of neighbors from model to do synchronous updating
            attitudes_others = np.array([self.model.hh_attitudes[hh.unique_id]
                                         for hh in self.neighbors])
            weights = [self.influence_weights[hh] for hh in self.neighbors]
            att_diff = np.sum(weights * (attitudes_others - self.attitude["PV"]))
            self.attitude["PV"] += self.model.influence_rate * att_diff
            self.attitude["no_PV"] = - self.attitude["PV"]

        # -- HOUSEHOLD DECISION MAKING -- #
        if not self.PV_installed:
            if self.savings > self.PV_costs:  # Income threshold
                if self.model.decision_making_model == "Rational":
                    U_PV, U_no_PV = self.NPV / self.model.max_NPV
                elif self.model.decision_making_model == "TPB":
                    U_PV, U_no_PV = self.TPB()

                # Compare utilities to decide on PV installation
                if U_PV > U_no_PV:
                    # Intention-behavior barrier:
                    if self.model.rng.binomial(1, 0.6):
                        self.PV_installed = True
                        self.savings -= self.PV_costs

