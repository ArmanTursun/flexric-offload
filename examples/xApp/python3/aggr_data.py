from collections import deque
import math

class AggrData():
    def __init__(self, maxlenth):
        self.maxqlenth = maxlenth
        self.aggr_bler = {
            'bler': deque(),       # save (bler, TS) tuple to queue
            'mean': 0,             # mean of bler
            'max': 0,              # max of bler in window
            'min': float('inf'),    # min of bler in window
            'num': 0,              # num of bler
            'sum': 0,              # sum of bler
            'sum_squares': 0,      # sum of squared bler
            'sum_cubes': 0         # sum of cubed deviations
        }
        self.aggr_enrg = {
            'enrg': deque(),       # save (energy, TS) tuple to queue
            'mean': 0,             # mean of energy
            'max': 0,              # max of energy in window
            'min': float('inf'),    # min of energy in window
            'num': 0,              # num of energy
            'sum': 0,              # sum of energy
            'sum_squares': 0,      # sum of squared energy
            'sum_cubes': 0         # sum of cubed deviations
        }
        
        # Deques to track min/max for bler and energy. Keeps add/pop in constant time
        self.bler_min_deque = deque()
        self.bler_max_deque = deque()
        self.enrg_min_deque = deque()
        self.enrg_max_deque = deque()

    def add_bler(self, bler_value, timestamp):
        """Add a new BLER value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_bler['bler']) >= self.maxqlenth:
            removed_bler, _ = self.aggr_bler['bler'].popleft()
            self.update_bler_removal(removed_bler)

        # Add new BLER value
        self.aggr_bler['bler'].append((bler_value, timestamp))
        self.update_bler_addition(bler_value)

    def update_bler_addition(self, bler_value):
        """Update BLER statistics when a new value is added"""
        # Update sums and counters
        self.aggr_bler['sum'] += bler_value
        self.aggr_bler['sum_squares'] += bler_value ** 2
        self.aggr_bler['num'] += 1
        
        # Update mean and other metrics
        self.aggr_bler['mean'] = self.aggr_bler['sum'] / self.aggr_bler['num']
        
        # Update sum of cubes (skewness component)
        mean = self.aggr_bler['mean']
        self.aggr_bler['sum_cubes'] += (bler_value - mean) ** 3

        # Update min and max deques for BLER
        while self.bler_min_deque and bler_value < self.aggr_bler['bler'][self.bler_min_deque[-1]][0]:
            self.bler_min_deque.pop()
        self.bler_min_deque.append(len(self.aggr_bler['bler']) - 1)

        while self.bler_max_deque and bler_value > self.aggr_bler['bler'][self.bler_max_deque[-1]][0]:
            self.bler_max_deque.pop()
        self.bler_max_deque.append(len(self.aggr_bler['bler']) - 1)

        self.aggr_bler['min'] = self.aggr_bler['bler'][self.bler_min_deque[0]][0]
        self.aggr_bler['max'] = self.aggr_bler['bler'][self.bler_max_deque[0]][0]

    def update_bler_removal(self, removed_bler):
        """Update BLER statistics when an old value is removed"""
        # Update sums and counters
        self.aggr_bler['sum'] -= removed_bler
        self.aggr_bler['sum_squares'] -= removed_bler ** 2
        self.aggr_bler['num'] -= 1
        
        if self.aggr_bler['num'] == 0:
            # Reset values if no data is left
            self.aggr_bler['mean'] = 0
            self.aggr_bler['sum_cubes'] = 0
            return
        
        # Update mean and sum of cubes (skewness component)
        mean = self.aggr_bler['sum'] / self.aggr_bler['num']
        self.aggr_bler['sum_cubes'] -= (removed_bler - mean) ** 3

        # Decrement indices in min/max deques
        if self.bler_min_deque and self.bler_min_deque[0] == 0:
            self.bler_min_deque.popleft()
        if self.bler_max_deque and self.bler_max_deque[0] == 0:
            self.bler_max_deque.popleft()

        self.bler_min_deque = deque([i - 1 for i in self.bler_min_deque])
        self.bler_max_deque = deque([i - 1 for i in self.bler_max_deque])

    def get_bler_skewness(self):
        """Calculate the skewness of the BLER data"""
        if self.aggr_bler['num'] < 3:
            return 0  # Skewness is undefined for fewer than 3 points
        
        n = self.aggr_bler['num']
        mean = self.aggr_bler['mean']
        variance = (self.aggr_bler['sum_squares'] / n) - (mean ** 2)
        if variance == 0:
            return 0  # No skewness if variance is zero

        skewness = (n / ((n - 1) * (n - 2))) * (self.aggr_bler['sum_cubes'] / (variance ** 1.5))
        return skewness

    def add_energy(self, energy_value, timestamp):
        """Add a new energy value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_enrg['enrg']) >= self.maxqlenth:
            removed_enrg, _ = self.aggr_enrg['enrg'].popleft()
            self.update_energy_removal(removed_enrg)

        # Add new energy value
        self.aggr_enrg['enrg'].append((energy_value, timestamp))
        self.update_energy_addition(energy_value)

    def update_energy_addition(self, energy_value):
        """Update energy statistics when a new value is added"""
        # Update sums and counters
        self.aggr_enrg['sum'] += energy_value
        self.aggr_enrg['sum_squares'] += energy_value ** 2
        self.aggr_enrg['num'] += 1
        
        # Update mean and other metrics
        self.aggr_enrg['mean'] = self.aggr_enrg['sum'] / self.aggr_enrg['num']
        
        # Update sum of cubes (skewness component)
        mean = self.aggr_enrg['mean']
        self.aggr_enrg['sum_cubes'] += (energy_value - mean) ** 3

        # Update min and max deques for energy
        while self.enrg_min_deque and energy_value < self.aggr_enrg['enrg'][self.enrg_min_deque[-1]][0]:
            self.enrg_min_deque.pop()
        self.enrg_min_deque.append(len(self.aggr_enrg['enrg']) - 1)

        while self.enrg_max_deque and energy_value > self.aggr_enrg['enrg'][self.enrg_max_deque[-1]][0]:
            self.enrg_max_deque.pop()
        self.enrg_max_deque.append(len(self.aggr_enrg['enrg']) - 1)

        self.aggr_enrg['min'] = self.aggr_enrg['enrg'][self.enrg_min_deque[0]][0]
        self.aggr_enrg['max'] = self.aggr_enrg['enrg'][self.enrg_max_deque[0]][0]

    def update_energy_removal(self, removed_enrg):
        """Update energy statistics when an old value is removed"""
        # Update sums and counters
        self.aggr_enrg['sum'] -= removed_enrg
        self.aggr_enrg['sum_squares'] -= removed_enrg ** 2
        self.aggr_enrg['num'] -= 1
        
        if self.aggr_enrg['num'] == 0:
            # Reset values if no data is left
            self.aggr_enrg['mean'] = 0
            self.aggr_enrg['sum_cubes'] = 0
            return
        
        # Update mean and sum of cubes (skewness component)
        mean = self.aggr_enrg['sum'] / self.aggr_enrg['num']
        self.aggr_enrg['sum_cubes'] -= (removed_enrg - mean) ** 3

        # Decrement indices in min/max deques
        if self.enrg_min_deque and self.enrg_min_deque[0] == 0:
            self.enrg_min_deque.popleft()
        if self.enrg_max_deque and self.enrg_max_deque[0] == 0:
            self.enrg_max_deque.popleft()

        self.enrg_min_deque = deque([i - 1 for i in self.enrg_min_deque])
        self.enrg_max_deque = deque([i - 1 for i in self.enrg_max_deque])

    def get_energy_skewness(self):
        """Calculate the skewness of the energy data"""
        if self.aggr_enrg['num'] < 3:
            return 0  # Skewness is undefined for fewer than 3 points
        
        n = self.aggr_enrg['num']
        mean = self.aggr_enrg['mean']
        variance = (self.aggr_enrg['sum_squares'] / n) - (mean ** 2)
        if variance == 0:
            return 0  # No skewness if variance is zero

        skewness = (n / ((n - 1) * (n - 2))) * (self.aggr_enrg['sum_cubes'] / (variance ** 1.5))
        return skewness

    def get_bler_stats(self, timestamp=None):
        """Get current BLER statistics including mean, max, min, and skewness."""
        if timestamp is None:
            #return {
            #    'mean': self.aggr_bler['mean'],
            #    'max': self.aggr_bler['max'],
            #    'min': self.aggr_bler['min'],
            #    'skewness': self.get_bler_skewness()
            #}
            return [self.aggr_bler['mean'], self.aggr_bler['max'], self.aggr_bler['min'], self.get_bler_skewness()]
        
        # Get current BLER statistics only if the first element's timestamp is later than the given timestamp
        if self.aggr_bler['bler'] and self.aggr_bler['bler'][0][1] > timestamp:
            #return {
            #    'mean': self.aggr_bler['mean'],
            #    'max': self.aggr_bler['max'],
            #    'min': self.aggr_bler['min'],
            #    'skewness': self.get_bler_skewness()
            #}
            return [self.aggr_bler['mean'], self.aggr_bler['max'], self.aggr_bler['min'], self.get_bler_skewness()]
        
        return None

    def get_energy_stats(self, timestamp=None):
        """Get current energy statistics including mean, max, min, and skewness."""
        if timestamp is None:
            #return {
            #    'mean': self.aggr_enrg['mean'],
            #    'max': self.aggr_enrg['max'],
            #    'min': self.aggr_enrg['min'],
            #    'skewness': self.get_energy_skewness()
            #}
            return [self.aggr_enrg['mean'], self.aggr_enrg['max'], self.aggr_enrg['min'], self.get_energy_skewness()]
        
        # Get current energy statistics only if the first element's timestamp is later than the given timestamp
        if self.aggr_enrg['enrg'] and self.aggr_enrg['enrg'][0][1] > timestamp:
            #return {
            #    'mean': self.aggr_enrg['mean'],
            #    'max': self.aggr_enrg['max'],
            #    'min': self.aggr_enrg['min'],
            #    'skewness': self.get_energy_skewness()
            #}
            return [self.aggr_enrg['mean'], self.aggr_enrg['max'], self.aggr_enrg['min'], self.get_energy_skewness()]
        
        return None
