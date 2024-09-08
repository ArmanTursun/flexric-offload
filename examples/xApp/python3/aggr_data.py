from collections import deque

#############################
#### Aggregated Data
#############################

class AggrData():
    def __init__(self, maxlenth):
        self.maxqlenth = maxlenth
        self.aggr_bler = {
            'bler': deque(),       # save (bler, TS) tuple to queue
            'mean': 0,             # mean of bler
            'max': 0,              # max of bler in window
            'min': float('inf'),    # min of bler in window
            'num': 0,              # num of bler
            'sum': 0               # sum of bler
        }
        self.aggr_enrg = {
            'enrg': deque(),       # save (energy, TS) tuple to queue
            'mean': 0,             # mean of energy
            'max': 0,              # max of energy in window
            'min': float('inf'),    # min of energy in window
            'num': 0,              # num of energy
            'sum': 0               # sum of energy
        }
        
        # Deques to track min/max for bler and energy. Keeps add/pop in contant time
        self.bler_min_deque = deque()
        self.bler_max_deque = deque()
        self.enrg_min_deque = deque()
        self.enrg_max_deque = deque()

    def add_bler(self, bler_value, timestamp):
        """Add a new BLER value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_bler['bler']) >= self.maxqlenth:
            removed_bler, _ = self.aggr_bler['bler'].popleft()
            self.aggr_bler['sum'] -= removed_bler
            self.aggr_bler['num'] -= 1
            if self.bler_min_deque and self.bler_min_deque[0] == 0:
                self.bler_min_deque.popleft()
            if self.bler_max_deque and self.bler_max_deque[0] == 0:
                self.bler_max_deque.popleft()

            # Decrement indices in min/max deques
            self.bler_min_deque = deque([i-1 for i in self.bler_min_deque])
            self.bler_max_deque = deque([i-1 for i in self.bler_max_deque])

        # Add new BLER value
        self.aggr_bler['bler'].append((bler_value, timestamp))
        self.aggr_bler['sum'] += bler_value
        self.aggr_bler['num'] += 1

        # Update the min/max deques for bler
        while self.bler_min_deque and bler_value < self.aggr_bler['bler'][self.bler_min_deque[-1]][0]:
            self.bler_min_deque.pop()
        self.bler_min_deque.append(len(self.aggr_bler['bler']) - 1)

        while self.bler_max_deque and bler_value > self.aggr_bler['bler'][self.bler_max_deque[-1]][0]:
            self.bler_max_deque.pop()
        self.bler_max_deque.append(len(self.aggr_bler['bler']) - 1)

        # Update the mean
        self.aggr_bler['mean'] = self.aggr_bler['sum'] / self.aggr_bler['num']
        self.aggr_bler['min'] = self.aggr_bler['bler'][self.bler_min_deque[0]][0]
        self.aggr_bler['max'] = self.aggr_bler['bler'][self.bler_max_deque[0]][0]

    def add_energy(self, energy_value, timestamp):
        """Add a new energy value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_enrg['enrg']) >= self.maxqlenth:
            removed_enrg, _ = self.aggr_enrg['enrg'].popleft()
            self.aggr_enrg['sum'] -= removed_enrg
            self.aggr_enrg['num'] -= 1
            if self.enrg_min_deque and self.enrg_min_deque[0] == 0:
                self.enrg_min_deque.popleft()
            if self.enrg_max_deque and self.enrg_max_deque[0] == 0:
                self.enrg_max_deque.popleft()

            # Decrement indices in min/max deques
            self.enrg_min_deque = deque([i-1 for i in self.enrg_min_deque])
            self.enrg_max_deque = deque([i-1 for i in self.enrg_max_deque])

        # Add new energy value
        self.aggr_enrg['enrg'].append((energy_value, timestamp))
        self.aggr_enrg['sum'] += energy_value
        self.aggr_enrg['num'] += 1

        # Update the min/max deques for energy
        while self.enrg_min_deque and energy_value < self.aggr_enrg['enrg'][self.enrg_min_deque[-1]][0]:
            self.enrg_min_deque.pop()
        self.enrg_min_deque.append(len(self.aggr_enrg['enrg']) - 1)

        while self.enrg_max_deque and energy_value > self.aggr_enrg['enrg'][self.enrg_max_deque[-1]][0]:
            self.enrg_max_deque.pop()
        self.enrg_max_deque.append(len(self.aggr_enrg['enrg']) - 1)

        # Update the mean
        self.aggr_enrg['mean'] = self.aggr_enrg['sum'] / self.aggr_enrg['num']
        self.aggr_enrg['min'] = self.aggr_enrg['enrg'][self.enrg_min_deque[0]][0]
        self.aggr_enrg['max'] = self.aggr_enrg['enrg'][self.enrg_max_deque[0]][0]

    def get_bler_stats(self, timestamp = None):
        if timestamp == None:
            return {
                'mean': self.aggr_bler['mean'],
                'max': self.aggr_bler['max'],
                'min': self.aggr_bler['min'],
                'num': self.aggr_bler['num'],
                'sum': self.aggr_bler['sum']
            }
        
        """Get current BLER statistics only if the first element's timestamp is later than the given timestamp"""
        if self.aggr_bler['bler'] and self.aggr_bler['bler'][0][1] > timestamp:
            return {
                'mean': self.aggr_bler['mean'],
                'max': self.aggr_bler['max'],
                'min': self.aggr_bler['min'],
                'num': self.aggr_bler['num'],
                'sum': self.aggr_bler['sum']
            }
        
        return None

    def get_energy_stats(self, timestamp = None):
        if timestamp == None:
            return {
                'mean': self.aggr_enrg['mean'],
                'max': self.aggr_enrg['max'],
                'min': self.aggr_enrg['min'],
                'num': self.aggr_enrg['num'],
                'sum': self.aggr_enrg['sum']
            }
        
        """Get current energy statistics only if the first element's timestamp is later than the given timestamp"""
        if self.aggr_enrg['enrg'] and self.aggr_enrg['enrg'][0][1] > timestamp:
            return {
                'mean': self.aggr_enrg['mean'],
                'max': self.aggr_enrg['max'],
                'min': self.aggr_enrg['min'],
                'num': self.aggr_enrg['num'],
                'sum': self.aggr_enrg['sum']
            }
        return None

