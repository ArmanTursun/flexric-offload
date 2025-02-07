from collections import deque
import math


class AggrData:
    def __init__(self, max_length, datasets):
        #self.max_length = max_length 
        self.datasets = datasets
        self.aggr_data = {
            dataset: {
                'data': deque(),  # save (value, timestamp) tuples
                'mean': 0,
                'var': 0,
                'max': float('-inf'),
                'min': float('inf'),
                'num': 0,
                'sum': 0,
                'sum_squares': 0,
                'sum_cubes': 0,  # for skewness calculation
                'min_deque': deque(),
                'max_deque': deque(), 
                'max_length': max_length if dataset == 'power' or dataset == 'ul_throughput' or dataset == 'ul_demand' or dataset == 'prate' else 1 #  or dataset == 'ul_prb' or dataset == 'ul_mcs'
            } for dataset in datasets
        }
        self.poor_sched_rate = 0.0

    def update_poor_sched_rate(self, poor_sched_rate):
        self.poor_sched_rate = poor_sched_rate

    def add_data(self, dataset, value, timestamp):
        """Add a new value for the specified dataset."""
        if dataset not in self.aggr_data:
            raise ValueError(f"Dataset '{dataset}' is not initialized.")

        #if value == 0:
        #    return 0
        data_queue = self.aggr_data[dataset]['data']
        max_length = self.aggr_data[dataset]['max_length']
        if len(data_queue) >= max_length:
            removed_value, _ = data_queue.popleft()
            self._update_removal(dataset, removed_value)

        data_queue.append((value, timestamp))
        self._update_addition(dataset, value)

    def _update_addition(self, dataset, value):
        """Update statistics for a dataset when a new value is added."""
        stats = self.aggr_data[dataset]
        old_mean = stats['mean']
        stats['sum'] += value
        stats['sum_squares'] += value ** 2
        stats['num'] += 1

        # Update mean
        stats['mean'] = stats['sum'] / stats['num']

        # Update variance using Welford's method
        if stats['num'] > 1:
            stats['var'] = ((stats['num'] - 2) / (stats['num'] - 1)) * stats['var'] + (value - old_mean) ** 2 / stats['num']

        # Update sum of cubes (for skewness calculation)
        mean = stats['mean']
        stats['sum_cubes'] += (value - mean) ** 3

        # Update min/max deques
        while stats['min_deque'] and value < stats['data'][stats['min_deque'][-1]][0]:
            stats['min_deque'].pop()
        stats['min_deque'].append(len(stats['data']) - 1)

        while stats['max_deque'] and value > stats['data'][stats['max_deque'][-1]][0]:
            stats['max_deque'].pop()
        stats['max_deque'].append(len(stats['data']) - 1)

        stats['min'] = stats['data'][stats['min_deque'][0]][0]
        stats['max'] = stats['data'][stats['max_deque'][0]][0]

    def _update_removal(self, dataset, removed_value):
        """Update statistics for a dataset when an old value is removed."""
        stats = self.aggr_data[dataset]
        old_mean = stats['mean']
        stats['sum'] -= removed_value
        stats['sum_squares'] -= removed_value ** 2
        stats['num'] -= 1

        if stats['num'] == 0:
            # Reset statistics if the dataset is empty
            stats.update({
                'mean': 0,
                'var': 0,
                'max': float('-inf'),
                'min': float('inf'),
                'sum': 0,
                'sum_squares': 0,
                'sum_cubes': 0,
                'min_deque': deque(),
                'max_deque': deque()
            })
            return

        # Update mean and variance using Welford's method
        stats['mean'] = stats['sum'] / stats['num']
        if stats['num'] > 1:
            stats['var'] = ((stats['num'] + 1) / stats['num']) * stats['var'] - ((removed_value - old_mean) ** 2) / stats['num']
        else:
            stats['var'] = 0
        
        # Update mean and sum of cubes
        mean = stats['mean']
        stats['sum_cubes'] -= (removed_value - mean) ** 3

        # Adjust indices in min/max deques
        if stats['min_deque'] and stats['min_deque'][0] == 0:
            stats['min_deque'].popleft()
        if stats['max_deque'] and stats['max_deque'][0] == 0:
            stats['max_deque'].popleft()

        stats['min_deque'] = deque([i - 1 for i in stats['min_deque']])
        stats['max_deque'] = deque([i - 1 for i in stats['max_deque']])

    def get_skewness(self, dataset):
        """Calculate skewness for the specified dataset."""
        if dataset not in self.aggr_data:
            raise ValueError(f"Dataset '{dataset}' is not initialized.")
        stats = self.aggr_data[dataset]
        if stats['num'] < 3:
            return 0  # Skewness is undefined for fewer than 3 points

        n = stats['num']
        mean = stats['mean']
        variance = (stats['sum_squares'] / n) - (mean ** 2)
        #variance = stats['var']
        if variance == 0:
            return 0  # No skewness if variance is zero

        skewness = (n / ((n - 1) * (n - 2))) * (stats['sum_cubes'] / (variance ** 1.5))
        return skewness

    def get_stats(self, dataset, timestamp=None):
        """Get current statistics for the specified dataset."""
        if dataset not in self.aggr_data:
            raise ValueError(f"Dataset '{dataset}' is not initialized.")
        stats = self.aggr_data[dataset]
        if timestamp is None:
            return {
                'mean': stats['mean'],
                'var': stats['var'],
                'max': stats['max'],
                'min': stats['min'],
                'skewness': self.get_skewness(dataset)
            }
        # Return statistics if the earliest timestamp in the queue is later than the given timestamp
        if stats['data'] and stats['data'][0][1] > timestamp:
            return {
                'mean': stats['mean'],
                'var': stats['var'],
                'max': stats['max'],
                'min': stats['min'],
                'skewness': self.get_skewness(dataset)
            }
        return None
