from typing import Optional

import numpy as np

from CADETProcess.processModel import ComponentSystem

# class UnitBase():
#      supported_bulk_reactions: set[ReactionBase] = {}
#      supported_particle_liquid_reactions: set[ReactionBase] = {}
#      supported_particle_solid_reactions: set[ReactionBase] = {}
#      supported_particle_cross_phase_reactions: set[ReactionBase] = {}

#      def __init__(self):
#          self._bulk_reactions = []
#          self._particle_liquid_reactions = []
#          self._particle_solid_reactions = []
#          self._particle_cross_phase_reactions = []

#      @property
#      def bulk_reactions(self) -> list[ReactionBase]:
#          return self.bulk_reactions

#      def add_bulk_reaction(self, reaction: ReactionBase):
#          if type(reaction) not in self.supported_bulk_reactions:
#              raise TypeError

#          self._bulk_reactions.append(reaction)

#      # TODO: Fill in reactions for other phases


# class GRM(UnitBase):
#      supported_bulk_reactions = {MAL, MM, Cryst}
#      supported_particle_liquid_reactions = {MAL, MM, Cryst}
#      supported_particle_solid_reactions = {MAL, MM}
#      supported_particle_cross_phase_reactions = {MAL}


# %% Demo

from CADETProcess.processModel.reaction_new import MichaelisMenten
from CADETProcess.processModel.reaction_new import CompetitiveInhibition, NonCompetitiveInhibition, UnCompetitiveInhibition
from CADETProcess.processModel.unitOperation import Cstr

component_set = ComponentSystem(3)


unit = Cstr(name="CSTR", component_system=component_set)


inhibition_1 = CompetitiveInhibition(component_system=component_set,
                                     substrate="1",
                                     inhibitors=["2", "3"],
                                     ki=[1.0, 2.0])

inhibition_2 = NonCompetitiveInhibition(component_system=component_set,
                                     substrate="1",
                                     inhibitors=["2", "3"],
                                     ki=[1.0, 2.0])

reaction_a = MichaelisMenten(component_system=component_set,
                            components=["1", "2"],
                            coefficients=[1.0, -1.0],
                            km=1.0,
                            vmax=1.0,
                            inhibition_reactions=inhibition_1)
reaction_b = MichaelisMenten()

unit.add_bulk_reaction(reaction_a)
unit.add_bulk_reaction(reaction_b)
