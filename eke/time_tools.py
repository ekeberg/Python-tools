"""Toosl dealing with time, such as timing a function and plotting a
progress bar."""
import time as _time
import datetime as _datetime
import sys as _sys
from collections import defaultdict as _defaultdict
import inspect as _inspect
import functools as _functools


class Progress(object):
    """Simple progress bar implementation"""
    def __init__(self, number_of_iterations, message=None, output_period=1.0):
        self._message = message
        self._number_of_iterations = number_of_iterations
        self._tasks_completed = 0
        self._time = _time.time()
        self._last_output_time = self._time
        self._last_iteration_time = 0.0
        self._expected_time_left = 0.0
        self._last_output_time = 0.0
        self._last_output_index = 0
        self._max_output_period = output_period
        self._done_char = "\u25a0"
        self._not_done_char = "\u25a1"

    def start(self):
        """Call before starting task."""
        self._tasks_completed = 0
        self._time = _time.time()
        self._last_output_time = self._time

    def finished(self):
        """Call when task is done."""
        self.print_message(always_output=True)
        _sys.stdout.write("\n")

    def iteration_completed(self, silent=False):
        """Call after every iteration. This function plots the otuput if a
        significant time passed since last plot.
        """
        self._tasks_completed += 1
        new_time = _time.time()
        self._last_iteration_time = new_time - self._time
        self._time = new_time
        self._expected_time_left = (self._last_iteration_time
                                    * (self._number_of_iterations -
                                       self._tasks_completed))
        if not silent:
            self.print_message()

    def print_message(self, always_output=False):
        new_time = self._time
        if always_output or new_time > (self._last_output_time
                                        + self._max_output_period):
            bar_length = 50
            done_length = int(float(self._tasks_completed)
                              / float(self._number_of_iterations)
                              * bar_length)
            not_done_length = bar_length - done_length

            message = f"{self._message}: " if self._message is not None else ""
            bar = (f"{self._done_char*done_length}"
                   f"{self._not_done_char*not_done_length} ")
            counter = (f"({self._tasks_completed}"
                       f"/{self._number_of_iterations} | ")
            ratio_value = self._tasks_completed/self._number_of_iterations
            percent = f"{100*ratio_value:.2f}%) "
            time = f"{int(self._expected_time_left)} seconds left"
            _sys.stdout.write("\r" + message + bar + counter + percent + time)
            _sys.stdout.flush()
            self._last_output_time = new_time
            self._last_output_index = self._tasks_completed

    @property
    def time_left(self):
        return self._expected_time_left


class StopWatch(object):
    """Uses wall time."""
    def __init__(self):
        self._running = False
        self._start_time = 0
        self._end_time = 0
        self._time_diff = 0

    def start(self):
        """Start watch"""
        self._running = True
        self._start_time = _time.time()

    def stop(self):
        """Stop and reset watch"""
        if not self._running:
            raise RuntimeError("Timer needs to be running to be stopped")
        self._running = False
        self._end_time = _time.time()
        self._time_diff = self._end_time - self._start_time

    def time(self):
        """The time in seconds that the clock was running."""
        if self._running:
            self._time_diff = _time.time() - self._start_time
        return self._time_diff

    def time_string(self):
        """Nicely formated string of the time the clock was running."""
        time_diff = self.time()
        if time_diff < 1.:
            return "{0} ms".format(time_diff*1000.)
        else:
            return str(_datetime.timedelta(seconds=time_diff))


class Timer:
    def __init__(self):
        self._records = _defaultdict(lambda: 0)
        self._start = _defaultdict(lambda: None)

    def start(self, name):
        self._start[name] = _time.time()  # type: ignore

    def stop(self, name):
        if self._start[name] is None:
            raise ValueError(f"Trying to stop inactive timer: {name}")
        self._records[name] += _time.time() - self._start[name]  # type: ignore
        self._start[name] = None

    def get_total(self):
        return self._records

    def print(self):
        print("Timing:")
        for n, v in self.get_total().items():
            print(f"{n}: {v}")


timer = Timer()


# def timed():
# def decorator(func):
def timed(func):
    func_signature = _inspect.signature(func)

    @_functools.wraps(func)
    def new_func(*args, **kwargs):
        # bound_arguments_no_default = func_signature.bind(*args, **kwargs)
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        args = bound_arguments.args
        
        timer.start(func.__name__)
        ret = func(*args)
        timer.stop(func.__name__)

        return ret
    return new_func
#     return decorator
