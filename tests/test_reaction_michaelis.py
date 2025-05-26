import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel.reaction_new import (
    MichaelisMenten, EnzyemeInhibtion
)


class Test_MichaelisMenten(unittest.TestCase):

    def create_simple_enzyme_reaction(self):
        # Enzyme reaction: S -> P
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')

        # Create Michaelis-Menten reaction
        reaction_model1 = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='simple_enzyme',
            km = [0.5],
            vmax = 2.0
        )


        return reaction_model1

    def create_competitive_inhibition_reaction(self):
        # Enzyme reaction with competitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('I')

        # Create Michaelis-Menten reaction
        reaction_model2 = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='competitive_inhibition',
            km=[0.5],
            vmax=2.0
        )


        # Add competitive inhibition
        inhibition = EnzyemeInhibtion(
            name = "CompInh",
            component_system=component_system,
            substrate='S',
            inhibitors= ['I'],
            competitive_rate=[0.1]
        )
        reaction_model2.add_inhibition_reaction(inhibition)

        return reaction_model2

    def create_uncompetitive_inhibition_reaction(self):
        # Enzyme reaction with uncompetitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('I')

        # Create Michaelis-Menten reaction
        reaction_model3 = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='uncompetitive_inhibition',
            km=[0.5],
            vmax=2.0
        )

        # Add uncompetitive inhibition
        inhibition = EnzyemeInhibtion(
            name="UnCompInh",
            component_system=component_system,
            substrate='S',
            inhibitors=['I'],
            uncompetitive_rate=[0.2]
        )
        reaction_model3.add_inhibition_reaction(inhibition)

        return reaction_model3

    def create_noncompetitive_inhibition_reaction(self):
        # Enzyme reaction with non-competitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('I')

        # Create Michaelis-Menten reaction
        reaction_model = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='noncompetitive_inhibition',
            km=[0.5],
            vmax=2.0
        )

        # Add non-competitive inhibition
        inhibition = EnzyemeInhibtion(
            name="NonCompInh",
            component_system=component_system,
            substrate='S',
            inhibitors=['I'],
            noncompetitive_rate=[0.3]
        )
        reaction_model.add_inhibition_reaction(inhibition)

        return reaction_model

    def test_simple_enzyme_reaction(self):
        reaction_model = self.create_simple_enzyme_reaction()

        # Test parameters
        self.assertEqual(reaction_model.km, [0.5])
        self.assertEqual(reaction_model.vmax, 2.0)
        self.assertEqual(reaction_model.name, 'simple_enzyme')
        self.assertEqual(len(reaction_model.inhibition_reactions), 0)

        # Test components and coefficients
        self.assertEqual(reaction_model.components, ['S', 'P'])
        self.assertEqual(reaction_model.coefficients, [-1, 1])


    def test_competitive_inhibition_reaction(self):
        reaction_model = self.create_competitive_inhibition_reaction()

        # Test parameters
        self.assertEqual(reaction_model.km, [0.5])
        self.assertEqual(reaction_model.vmax, 2.0)
        self.assertEqual(reaction_model.name, 'competitive_inhibition')

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, EnzyemeInhibtion)
        self.assertEqual(inhibition.competitive_rate, [0.1])
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, ['I'])


    def test_uncompetitive_inhibition_reaction(self):
        reaction_model = self.create_uncompetitive_inhibition_reaction()

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, EnzyemeInhibtion)
        self.assertEqual(inhibition.uncompetitive_rate, [0.2])
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, ['I'])


    def test_noncompetitive_inhibition_reaction(self):
        reaction_model = self.create_noncompetitive_inhibition_reaction()

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, EnzyemeInhibtion)
        self.assertEqual(inhibition.noncompetitive_rate, [0.3])
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, ['I'])

    def test_multiple_inhibitors(self):
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('Inh1')
        component_system.add_component('Inh2')

        # Create Michaelis-Menten reaction
        reaction_model = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='multiple_inhibitors',
            km=[0.5],
            vmax=2.0
        )

        # Add inhibition with multiple inhibitors
        inhibition = EnzyemeInhibtion(
            name="MultiInh",
            component_system=component_system,
            substrate='S',
            inhibitors=['Inh1', 'Inh2'],
            competitive_rate=[0.1,0.2]
        )
        reaction_model.add_inhibition_reaction(inhibition)

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        np.testing.assert_array_equal(inhibition.inhibitors, np.array(['Inh1', 'Inh2']))
        np.testing.assert_array_equal(inhibition.competitive_rate, np.array([0.1, 0.2]))


if __name__ == '__main__':
    unittest.main()
