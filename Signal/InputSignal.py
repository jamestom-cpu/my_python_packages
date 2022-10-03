import numpy as np
import pandas as pd

class Signal(object):
  signal_id = -1

  def __init__(self, name="", initial_value=0, low=0, high=1, slew=1e-3):
    Signal.signal_id = Signal.signal_id + 1
    self.id = Signal.signal_id

    if name:
      self.name = name
    else:
      self.name = "signal-" + str(self.signal_id)

    self.data = [
        [0, initial_value]
    ]

    self.slew = slew
    self.at_time = 0

  def value_at(self, time):
    previous_point = None

    for point in self.data:
      if point[0] == time:
        return point[1]
      elif point[0] > time:
        print("here")

      previous_point = point

    return previous_point[1]

  def end_time(self):
    return self.data[-1][0]

  def at(self, time):
    self.at_time = time
    return self

  def to(self, value):
    self.data.append([self.at_time, self.value_at(self.at_time)])

    self.at_time = self.at_time + self.slew
    self.data.append([self.at_time, value])

    return self

  def extend_to(self, time):
    if self.end_time() < time:
      self.data.append([time, self.value_at(self.end_time())])


  def save(self):
    file_name = self.name + ".pwl"
    with open(file_name, "w") as f:
      for point in self.data:
        f.write(str(point[0]) + "\t" + str(point[1]) + "\n")

  def as_lists(self):
    return tuple([list(t) for t in zip(*self.data)])

  def scale_time(self, factor):
    self.data = [[point[0]*factor, point[1]] for point in self.data]

  def scale_value(self, factor):
    self.data = [[point[0], point[1]*factor] for point in self.data]






## Create my own class with extra functions
class MySignal(Signal):
    def __init__(self, name="", initial_value=0, low=0, high=1, slew=1e-3):
        super().__init__(name="", initial_value=0, low=0, high=1, slew=1e-3)
    
    def to_numpy(self):
        self.data_np = np.array(self.data)
        return self.data_np
    def get_time(self):
        data_np = self.to_numpy()
        return self.data_np[:,0]
    def get_values(self):
        data_np = self.to_numpy()
        return self.data_np[:,1]
    def repeat(self, N):
        assert N > 1, "N is the number of periods"
        values = self.get_values()
        time = self.get_time()

        for period in range(N-1):
            new_time = time+self.end_time()
            new_data = np.stack([new_time, values], axis=1)
            
            self.data += new_data.tolist()
        self.data = np.unique(self.data, axis=0).tolist()
        return self.data
    
    def make_pandas_table(self):
        table = pd.DataFrame(self.data, columns=["time", "value"]).set_index("time")
        return table

    def from_func(self, func, time):
        self.data = get_PWL_from_function(func, time)
        return self.data



def get_PWL_from_function(func, time):
    values = func(time)
    data = [[t, value] for (t, value) in zip(time, values)]
    return data
