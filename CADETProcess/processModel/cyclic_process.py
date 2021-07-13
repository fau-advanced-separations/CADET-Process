class CyclicProcess(Process):
    n_cycles = UnsignedInteger(default=1)
    
    @property
    def _time_complete(self):
        """Defines the complete time for simulation.

        First the number of cycles is set for evaluating the complete time of
        simulation by multiplication the value of the cycle time with the
        number of cycles. To get the number of steps for the linspace without
        start and ending point the indices are evaluated.

        Returns
        -------
        time_complete : ndarray
            array of the time vector with 1 decimal-point rounded values from
            zero to time complete with number of indices as steps.
        """
        complete_time = self.n_cycles * self.cycle_time
        indices = self.n_cycles*math.ceil(self.cycle_time) - (self.n_cycles-1)
        return np.round(np.linspace(0, complete_time, indices), 1)