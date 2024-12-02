from collections import deque
import math

class AggrData():
    def __init__(self, maxlenth):
        self.maxqlenth = maxlenth
        self.aggr_data1 = {
            'data1': deque(),       # save (data1, TS) tuple to queue
            'mean': 0,             # mean of data1
            'max': 0,              # max of data1 in window
            'min': float('inf'),    # min of data1 in window
            'num': 0,              # num of data1
            'sum': 0,              # sum of data1
            'sum_squares': 0,      # sum of squared data1
            'sum_cubes': 0         # sum of cubed deviations
        }
        self.aggr_data2 = {
            'data2': deque(),       # save (data2, TS) tuple to queue
            'mean': 0,             # mean of data2
            'max': 0,              # max of data2 in window
            'min': float('inf'),    # min of data2 in window
            'num': 0,              # num of data2
            'sum': 0,              # sum of data2
            'sum_squares': 0,      # sum of squared data2
            'sum_cubes': 0         # sum of cubed deviations
        }
        
        # Deques to track min/max for data1 and data2. Keeps add/pop in constant time
        self.data1_min_deque = deque()
        self.data1_max_deque = deque()
        self.data2_min_deque = deque()
        self.data2_max_deque = deque()

    def add_data1(self, data1_value, timestamp):
        """Add a new data1 value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_data1['data1']) >= self.maxqlenth:
            removed_data1, _ = self.aggr_data1['data1'].popleft()
            self.update_data1_removal(removed_data1)

        # Add new data1 value
        self.aggr_data1['data1'].append((data1_value, timestamp))
        self.update_data1_addition(data1_value)

    def update_data1_addition(self, data1_value):
        """Update data1 statistics when a new value is added"""
        # Update sums and counters
        self.aggr_data1['sum'] += data1_value
        self.aggr_data1['sum_squares'] += data1_value ** 2
        self.aggr_data1['num'] += 1
        
        # Update mean and other metrics
        self.aggr_data1['mean'] = self.aggr_data1['sum'] / self.aggr_data1['num']
        
        # Update sum of cubes (skewness component)
        mean = self.aggr_data1['mean']
        self.aggr_data1['sum_cubes'] += (data1_value - mean) ** 3

        # Update min and max deques for data1
        while self.data1_min_deque and data1_value < self.aggr_data1['data1'][self.data1_min_deque[-1]][0]:
            self.data1_min_deque.pop()
        self.data1_min_deque.append(len(self.aggr_data1['data1']) - 1)

        while self.data1_max_deque and data1_value > self.aggr_data1['data1'][self.data1_max_deque[-1]][0]:
            self.data1_max_deque.pop()
        self.data1_max_deque.append(len(self.aggr_data1['data1']) - 1)

        self.aggr_data1['min'] = self.aggr_data1['data1'][self.data1_min_deque[0]][0]
        self.aggr_data1['max'] = self.aggr_data1['data1'][self.data1_max_deque[0]][0]

    def update_data1_removal(self, removed_data1):
        """Update data1 statistics when an old value is removed"""
        # Update sums and counters
        self.aggr_data1['sum'] -= removed_data1
        self.aggr_data1['sum_squares'] -= removed_data1 ** 2
        self.aggr_data1['num'] -= 1
        
        if self.aggr_data1['num'] == 0:
            # Reset values if no data is left
            self.aggr_data1['mean'] = 0
            self.aggr_data1['sum_cubes'] = 0
            return
        
        # Update mean and sum of cubes (skewness component)
        mean = self.aggr_data1['sum'] / self.aggr_data1['num']
        self.aggr_data1['sum_cubes'] -= (removed_data1 - mean) ** 3

        # Decrement indices in min/max deques
        if self.data1_min_deque and self.data1_min_deque[0] == 0:
            self.data1_min_deque.popleft()
        if self.data1_max_deque and self.data1_max_deque[0] == 0:
            self.data1_max_deque.popleft()

        self.data1_min_deque = deque([i - 1 for i in self.data1_min_deque])
        self.data1_max_deque = deque([i - 1 for i in self.data1_max_deque])

    def get_data1_skewness(self):
        """Calculate the skewness of the data1 data"""
        if self.aggr_data1['num'] < 3:
            return 0  # Skewness is undefined for fewer than 3 points
        
        n = self.aggr_data1['num']
        mean = self.aggr_data1['mean']
        variance = (self.aggr_data1['sum_squares'] / n) - (mean ** 2)
        if variance == 0:
            return 0  # No skewness if variance is zero

        skewness = (n / ((n - 1) * (n - 2))) * (self.aggr_data1['sum_cubes'] / (variance ** 1.5))
        return skewness

    def add_data2(self, data2_value, timestamp):
        """Add a new data2 value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_data2['data2']) >= self.maxqlenth:
            removed_data2, _ = self.aggr_data2['data2'].popleft()
            self.update_data2_removal(removed_data2)

        # Add new data2 value
        self.aggr_data2['data2'].append((data2_value, timestamp))
        self.update_data2_addition(data2_value)

    def update_data2_addition(self, data2_value):
        """Update data2 statistics when a new value is added"""
        # Update sums and counters
        self.aggr_data2['sum'] += data2_value
        self.aggr_data2['sum_squares'] += data2_value ** 2
        self.aggr_data2['num'] += 1
        
        # Update mean and other metrics
        self.aggr_data2['mean'] = self.aggr_data2['sum'] / self.aggr_data2['num']
        
        # Update sum of cubes (skewness component)
        mean = self.aggr_data2['mean']
        self.aggr_data2['sum_cubes'] += (data2_value - mean) ** 3

        # Update min and max deques for data2
        while self.data2_min_deque and data2_value < self.aggr_data2['data2'][self.data2_min_deque[-1]][0]:
            self.data2_min_deque.pop()
        self.data2_min_deque.append(len(self.aggr_data2['data2']) - 1)

        while self.data2_max_deque and data2_value > self.aggr_data2['data2'][self.data2_max_deque[-1]][0]:
            self.data2_max_deque.pop()
        self.data2_max_deque.append(len(self.aggr_data2['data2']) - 1)

        self.aggr_data2['min'] = self.aggr_data2['data2'][self.data2_min_deque[0]][0]
        self.aggr_data2['max'] = self.aggr_data2['data2'][self.data2_max_deque[0]][0]

    def update_data2_removal(self, removed_data2):
        """Update data2 statistics when an old value is removed"""
        # Update sums and counters
        self.aggr_data2['sum'] -= removed_data2
        self.aggr_data2['sum_squares'] -= removed_data2 ** 2
        self.aggr_data2['num'] -= 1
        
        if self.aggr_data2['num'] == 0:
            # Reset values if no data is left
            self.aggr_data2['mean'] = 0
            self.aggr_data2['sum_cubes'] = 0
            return
        
        # Update mean and sum of cubes (skewness component)
        mean = self.aggr_data2['sum'] / self.aggr_data2['num']
        self.aggr_data2['sum_cubes'] -= (removed_data2 - mean) ** 3

        # Decrement indices in min/max deques
        if self.data2_min_deque and self.data2_min_deque[0] == 0:
            self.data2_min_deque.popleft()
        if self.data2_max_deque and self.data2_max_deque[0] == 0:
            self.data2_max_deque.popleft()

        self.data2_min_deque = deque([i - 1 for i in self.data2_min_deque])
        self.data2_max_deque = deque([i - 1 for i in self.data2_max_deque])

    def get_data2_skewness(self):
        """Calculate the skewness of the data2 data"""
        if self.aggr_data2['num'] < 3:
            return 0  # Skewness is undefined for fewer than 3 points
        
        n = self.aggr_data2['num']
        mean = self.aggr_data2['mean']
        variance = (self.aggr_data2['sum_squares'] / n) - (mean ** 2)
        if variance == 0:
            return 0  # No skewness if variance is zero

        skewness = (n / ((n - 1) * (n - 2))) * (self.aggr_data2['sum_cubes'] / (variance ** 1.5))
        return skewness

    def get_data1_stats(self, timestamp=None):
        """Get current data1 statistics including mean, max, min, and skewness."""
        if timestamp is None:
            #return {
            #    'mean': self.aggr_data1['mean'],
            #    'max': self.aggr_data1['max'],
            #    'min': self.aggr_data1['min'],
            #    'skewness': self.get_data1_skewness()
            #}
            return [self.aggr_data1['mean'], self.aggr_data1['max'], self.aggr_data1['min'], self.get_data1_skewness()]
        
        # Get current data1 statistics only if the first element's timestamp is later than the given timestamp
        if self.aggr_data1['data1'] and self.aggr_data1['data1'][0][1] > timestamp:
            #return {
            #    'mean': self.aggr_data1['mean'],
            #    'max': self.aggr_data1['max'],
            #    'min': self.aggr_data1['min'],
            #    'skewness': self.get_data1_skewness()
            #}
            return [self.aggr_data1['mean'], self.aggr_data1['max'], self.aggr_data1['min'], self.get_data1_skewness()]
        
        return None

    def get_data2_stats(self, timestamp=None):
        """Get current data2 statistics including mean, max, min, and skewness."""
        if timestamp is None:
            #return {
            #    'mean': self.aggr_data2['mean'],
            #    'max': self.aggr_data2['max'],
            #    'min': self.aggr_data2['min'],
            #    'skewness': self.get_data2_skewness()
            #}
            return [self.aggr_data2['mean'], self.aggr_data2['max'], self.aggr_data2['min'], self.get_data2_skewness()]
        
        # Get current data2 statistics only if the first element's timestamp is later than the given timestamp
        if self.aggr_data2['data2'] and self.aggr_data2['data2'][0][1] > timestamp:
            #return {
            #    'mean': self.aggr_data2['mean'],
            #    'max': self.aggr_data2['max'],
            #    'min': self.aggr_data2['min'],
            #    'skewness': self.get_data2_skewness()
            #}
            return [self.aggr_data2['mean'], self.aggr_data2['max'], self.aggr_data2['min'], self.get_data2_skewness()]
        
        return None
