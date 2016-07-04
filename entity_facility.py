__author__ = 'vinay_vijayan'

# TODO remove hacks on random variate generation
# TODO work on speed ups, is a pandas df slowing things?



import pandas as pd
import numpy as np

LIFE_OF_TOILET = 5
MAX_USES = 80
NUMBER_SIMULATIONS = 1825
BUY_OUT_PERCENT = 0.10


class EntityFacility(object):

    def simulate(self, time_window=5, dict_start_times={'centre': 0, 'north': 2, 'south': 3, 'east': 4, 'west': 5}):

        self.dict_start_times = dict_start_times
        self.time_window = time_window
        data = self._bivariate_custom()
        self.df_time_demand = pd.DataFrame(data=data, columns=['time', 'demand'])
        self.df_time_demand['time'] = self.dict_start_times['centre'] + ((self.df_time_demand['time']) * LIFE_OF_TOILET)
        self.df_time_demand['demand'] *= MAX_USES
        self.df_time_demand = self.df_time_demand.sort_values(by='time').reset_index()

        for index, row in self.df_time_demand.iterrows():
            number_reduction = len([each for each in self.dict_start_times if each != 'centre' and row['time']
                                    >= self.dict_start_times[each]])
            if number_reduction > 0:
                self.df_time_demand.set_value(index, 'demand', row['demand'] * (1-(number_reduction * BUY_OUT_PERCENT)))

    # very crude method to generate positively correlated bivariates of time and demand
    # TODO This is very hacky, read lit on rv gen that has constantly varying parameters
    def _bivariate_custom(self):
        data = []
        for i in range(0, NUMBER_SIMULATIONS):
            a = float(i)/NUMBER_SIMULATIONS
            b = 1 - (a + 0.00001)
            data.append([a, np.random.beta(a+0.00001, b, 1)[0]])
        return data

    # TODO This is very hacky, read lit on rv gen that has constantly varying parameters
    def _get_installation_cost(self):
        start_time = self.dict_start_times['centre']
        if start_time < self.time_window:
            mean = 500 - np.random.uniform(0, 1) * (start_time/self.time_window) * 100
            if mean <= 0:
                mean = 0.01
            std_dev = 20 * (start_time/self.time_window)
            if std_dev <= 0:
                std_dev = 0.01
            installation_cost = np.random.normal(mean, std_dev)
        else:
            installation_cost = 0
        return installation_cost

    def get_stats(self):
        return self.df_time_demand['demand'].sum(), self._get_installation_cost()

if __name__ == '__main__':
    test_object = EntityFacility()
    test_object.simulate()
    number_served, cost = test_object.get_stats()

