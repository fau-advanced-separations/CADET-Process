import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel.reaction_new import (
    MichaelisMenten,
    CompetitiveInhibition,
    UnCompetitiveInhibition,
    NonCompetitiveInhibition,
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
            name='simple_enzyme'
        )
        # Set Michaelis-Menten parameters
        reaction_model1.km = [0.5]
        reaction_model1.vmax = 2.0

        return reaction_model1

    def create_competitive_inhibition_reaction(self):
        # Enzyme reaction with competitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('Inhibitor')

        # Create Michaelis-Menten reaction
        reaction_model2 = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='competitive_inhibition'
        )
        # Set Michaelis-Menten parameters
        reaction_model2.km = [0.5]
        reaction_model2.vmax = 2.0

        # Add competitive inhibition
        inhibition = CompetitiveInhibition(
            type="Competitive Inhibition",
            component_system=component_system,
            substrate='S',
            inhibitors='Inhibitor',
            ki=0.1
        )
        reaction_model2.add_inhibition_reaction(inhibition)

        return reaction_model2

    def create_uncompetitive_inhibition_reaction(self):
        # Enzyme reaction with uncompetitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('Inhibitor')

        # Create Michaelis-Menten reaction
        reaction_model3 = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='uncompetitive_inhibition'
        )
        # Set Michaelis-Menten parameters
        reaction_model3.km = [0.5]
        reaction_model3.vmax = 2.0

        # Add uncompetitive inhibition
        inhibition = UnCompetitiveInhibition(
            type="Uncompetitive Inhibition",
            component_system=component_system,
            substrate='S',
            inhibitors='Inhibitor',
            ki=0.2
        )
        reaction_model3.add_inhibition_reaction(inhibition)

        return reaction_model3

    def create_noncompetitive_inhibition_reaction(self):
        # Enzyme reaction with non-competitive inhibition: S -> P, inhibited by I
        component_system = ComponentSystem()
        component_system.add_component('S')
        component_system.add_component('P')
        component_system.add_component('Inhibitor')

        # Create Michaelis-Menten reaction
        reaction_model = MichaelisMenten(
            component_system=component_system,
            components=['S', 'P'],
            coefficients=[-1, 1],
            name='noncompetitive_inhibition'
        )
        # Set Michaelis-Menten parameters
        reaction_model.km = [0.5]
        reaction_model.vmax = 2.0

        # Add non-competitive inhibition
        inhibition = NonCompetitiveInhibition(
            type="Non-competitive Inhibition",
            component_system=component_system,
            substrate='S',
            inhibitors='Inhibitor',
            ki=0.3
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

        # Test string representation
        self.assertIn("Michaelis-Menten without inhibition", str(reaction_model))

    def test_competitive_inhibition_reaction(self):
        reaction_model = self.create_competitive_inhibition_reaction()

        # Test parameters
        self.assertEqual(reaction_model.km, [0.5])
        self.assertEqual(reaction_model.vmax, 2.0)
        self.assertEqual(reaction_model.name, 'competitive_inhibition')

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, CompetitiveInhibition)
        self.assertEqual(inhibition.ki, 0.1)
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, 'Inhibitor')
        self.assertEqual(inhibition.type, "Competitive Inhibition")

        # Test string representation
        self.assertIn("Michaelis-Menten with Competitive Inhibition", str(reaction_model))

    def test_uncompetitive_inhibition_reaction(self):
        reaction_model = self.create_uncompetitive_inhibition_reaction()

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, UnCompetitiveInhibition)
        self.assertEqual(inhibition.ki, 0.2)
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, 'Inhibitor')
        self.assertEqual(inhibition.type, "Uncompetitive Inhibition")

        # Test string representation
        self.assertIn("Michaelis-Menten with Uncompetitive Inhibition", str(reaction_model))

    def test_noncompetitive_inhibition_reaction(self):
        reaction_model = self.create_noncompetitive_inhibition_reaction()

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        self.assertIsInstance(inhibition, NonCompetitiveInhibition)
        self.assertEqual(inhibition.ki, 0.3)
        self.assertEqual(inhibition.substrate, 'S')
        self.assertEqual(inhibition.inhibitors, 'Inhibitor')
        self.assertEqual(inhibition.type, "Non-competitive Inhibition")

        # Test string representation
        self.assertIn("Michaelis-Menten with Non-competitive Inhibition", str(reaction_model))

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
            name='multiple_inhibitors'
        )
        reaction_model.km = [0.5]
        reaction_model.vmax = 2.0

        # Add inhibition with multiple inhibitors
        inhibition = CompetitiveInhibition(
            type="Competitive Inhibition",
            component_system=component_system,
            substrate='S',
            inhibitors=['Inh1', 'Inh2'],
            ki=[0.1, 0.2]
        )
        reaction_model.add_inhibition_reaction(inhibition)

        # Test inhibition
        self.assertEqual(len(reaction_model.inhibition_reactions), 1)
        inhibition = reaction_model.inhibition_reactions[0]
        np.testing.assert_array_equal(inhibition.inhibitors, np.array(['Inh1', 'Inh2']))
        np.testing.assert_array_equal(inhibition.ki, np.array([0.1, 0.2]))


if __name__ == '__main__':
    unittest.main()
