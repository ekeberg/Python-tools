import time
import datetime

class Progress:
    def __init__(self, message, number_of_iterations, output_period = 1.0):
        self._message = message
        self._number_of_iterations = number_of_iterations
        self._tasks_completed = 0
        self._time = time.clock()
        self._last_iteration_time = 0.0
        self._expected_time_left = 0.0
        self._last_output_time = 0.0
        self._last_output_index = 0
        self._max_output_period = output_period

    def start(self):
        self._tasks_completed = 0
        self._time = time.clock()
        self._last_output_time = time.clock()

    def iteration_completed(self):
        self._tasks_completed += 1
        new_time = time.clock()
        self._last_iteration_time = new_time - self._time
        self._time = new_time
        self._expected_time_left = self._last_iteration_time * (self._number_of_iterations -
                                                                self._tasks_completed)
        if (new_time > (self._last_output_time + self._max_output_period)):
            print "%s: Iteration tock %g seconds. Expected %g seconds left" % (self._message,
                                                                               self._last_iteration_time,
                                                                               self._expected_time_left)
            self._last_output_time = new_time
            self._last_output_index = self._tasks_completed
        

class StopWatch:
    def __init__(self):
        self._running = False
        self.time_diff = 0.

    def start(self):
        self._running = True
        self._start_time = time.time()

    def stop(self):
        if not self._running:
            raise StateError("Timer needs to be running to be stopped")
        self._end_time = time.time()
        self._time_diff = self._end_time - self._start_time

    def str(self):
        if self._time_diff < 1.:
            return "%g ms" % (self._time_diff*1000.)
        else:
            return str(datetime.timedelta(seconds=self._time_diff))

        
