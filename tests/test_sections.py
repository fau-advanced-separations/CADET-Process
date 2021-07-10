import unittest

from scipy import integrate

class TestTimeLine(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def create_time_line_1(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,1,(1.5,0))
        section_1 = CADETProcess.common.Section(1,2,(0,0))
        section_2 = CADETProcess.common.Section(2,3,(0,1))
        section_3 = CADETProcess.common.Section(3,4,(1,0))
        section_4 = CADETProcess.common.Section(4,5,(2,0))
        section_5 = CADETProcess.common.Section(5,6,(2,-2))

        time_line_1 = CADETProcess.common.TimeLine()
        time_line_1.add_section(section_0)
        time_line_1.add_section(section_1)
        time_line_1.add_section(section_2)
        time_line_1.add_section(section_3)
        time_line_1.add_section(section_4)
        time_line_1.add_section(section_5)

        return time_line_1

    def test_time_line_1(self):
        """Test concentration profile with steps and gradients"""
        time_line_1 = self.create_time_line_1()
        self.assertEqual(time_line_1.value(0), 1.5)
        self.assertEqual(time_line_1.value(1), 0.0)
        self.assertEqual(time_line_1.value(2), 0.0)
        self.assertEqual(time_line_1.value(3), 1.0)
        self.assertEqual(time_line_1.value(4), 2.0)
        self.assertEqual(time_line_1.value(5), 2.0)
        self.assertEqual(time_line_1.value(5.5), 1.0)
        self.assertEqual(time_line_1.value(6), 0.0)


    def create_time_line_2_Q(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,1,1)
        section_1 = CADETProcess.common.Section(1,2,0)

        time_line_2_Q = CADETProcess.common.TimeLine()
        time_line_2_Q.add_section(section_0)
        time_line_2_Q.add_section(section_1)

        return time_line_2_Q

    def create_time_line_2_c(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,2,1)

        time_line_2_c = CADETProcess.common.TimeLine()
        time_line_2_c.add_section(section_0)

        return time_line_2_c

    def test_time_line_2(self):
        """Feed situation: constant concentration and variable flow"""
        time_line_2_Q = self.create_time_line_2_Q()
        time_line_2_c = self.create_time_line_2_c()

        integral = integrate.quad(
            lambda t: time_line_2_Q.value(t) * time_line_2_c.value(t), 0,2
        )[0]

        self.assertEqual(integral, 1.0)


    def create_time_line_3_Q(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,1,1)
        section_1 = CADETProcess.common.Section(1,2,0)
        section_2 = CADETProcess.common.Section(2,3,1)

        time_line_3_Q = CADETProcess.common.TimeLine()
        time_line_3_Q.add_section(section_0)
        time_line_3_Q.add_section(section_1)
        time_line_3_Q.add_section(section_2)

        return time_line_3_Q

    def create_time_line_3_c(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,0.5,0)
        section_1 = CADETProcess.common.Section(0.5,2.5,1)
        section_2 = CADETProcess.common.Section(2.5,3,0)

        time_line_3_c = CADETProcess.common.TimeLine()
        time_line_3_c.add_section(section_0)
        time_line_3_c.add_section(section_1)
        time_line_3_c.add_section(section_2)

        return time_line_3_c

    def test_time_line_3(self):
        """Recycle situation: constant concentration with intermediate zero-flow
        """
        time_line_3_Q = self.create_time_line_3_Q()
        time_line_3_c = self.create_time_line_3_c()

        integral = integrate.quad(lambda t:
        time_line_3_Q.value(t) * time_line_3_c.value(t), 0,3)[0]
        self.assertAlmostEqual(integral, 1.0)


    def create_time_line_4_Q(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,1,(0,1))
        section_1 = CADETProcess.common.Section(1,2,0)

        time_line_4_Q = CADETProcess.common.TimeLine()
        time_line_4_Q.add_section(section_0)
        time_line_4_Q.add_section(section_1)

        return time_line_4_Q

    def create_time_line_4_c(self):
        import CADETProcess

        section_0 = CADETProcess.common.Section(0,2,1)

        time_line_4_c = CADETProcess.common.TimeLine()
        time_line_4_c.add_section(section_0)

        return time_line_4_c

    def test_time_line_4(self):
        """Gradient situation: constant concentration with variable flow rate.
        """
        time_line_4_Q = self.create_time_line_4_Q()
        time_line_4_c = self.create_time_line_4_c()

        integral = integrate.quad(lambda t:
        time_line_4_Q.value(t) * time_line_4_c.value(t), 0,2)[0]

        self.assertEqual(integral, 0.5)


if __name__ == '__main__':
    unittest.main()