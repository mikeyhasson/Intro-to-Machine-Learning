#################################
# Your name: Michael Hasson
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math



class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """

        X = np.sort(np.random.uniform(size=m))
        Y= [np.random.choice([0.0, 1.0], p=[0.9, 0.1]) if (0.2<x<0.4 or 0.6<x<0.8)
            else np.random.choice([0.0, 1.0], p=[0.2, 0.8]) for x in X]

        return np.column_stack((X,Y))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_range = np.arange(m_first, m_last + 1, step)
        arr = np.array([self.get_errors_by_size(size,k,T) for size in m_range])
        plt.plot(m_range,arr[:, 0], label='Empirical Error')
        plt.plot(m_range,arr[:, 1], label='True Error')
        plt.legend()
        plt.xlabel("m")
        plt.show()
        return arr

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        k_range, emp_err = self.experiment_k_range_erm_no_show(m, k_first, k_last, step)
        plt.show()
        return k_range[np.argmin(emp_err)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        delta=0.1
        k_range,emp_error = self.experiment_k_range_erm_no_show(m, k_first, k_last, step)
        penalty_per_k =np.sqrt((2 * k_range + math.log(2/delta,math.e))/m) #VC_dim is 2k, from the Ex
        plt.plot(k_range,penalty_per_k,label='Penalty')
        sum_emp_err_and_penalty = penalty_per_k + emp_error
        plt.plot(k_range,sum_emp_err_and_penalty,label='Sum of Penalty and Empirical Error')
        plt.show()

        return k_range[np.argmin(sum_emp_err_and_penalty)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        count_holdout=int(m/5)
        holdout = sample[:count_holdout] # 20% validation
        train = sample[count_holdout:]
        train =  train[train[:, 0].argsort()]
        train_X, train_Y= train[:, 0], train[:, 1]
        G = [intervals.find_best_interval(train_X, train_Y, k)[0] for k in range(1, 11)]
        emp_err_G_hold = [sum([self.zo_loss(sample[0],sample[1],g) for sample in holdout])/count_holdout for g in G]
        print("best k:", np.argmin(emp_err_G_hold) + 1)
        print("best hypotesis:", G[np.argmin(emp_err_G_hold)])
        print("Empirical Error over the holdout set:", np.min(emp_err_G_hold))
        return np.argmin(emp_err_G_hold) + 1

    #################################
    # Place for additional methods
    #################################
    def get_errors_by_size(self, size, k, T):
        empirical_error = 0
        true_error = 0
        for j in range(T):
            sample = self.sample_from_D(size)
            sample_X = sample[:, 0]
            sample_Y = sample[:, 1]
            interval_set, error = intervals.find_best_interval(sample_X, sample_Y, k)
            empirical_error += error
            true_error += self.calc_true_error(interval_set)
        empirical_error = empirical_error / (T * size)  # divide by size to get emp-err and by T since we summed T expr.
        true_error = true_error / T  # summed T true_errors, getting mean.
        return np.array([empirical_error, true_error])

    def calc_true_error(self, h_intervals):
        # calculations in pdf
        # ep(h)=0.2*|I∩ I_1|+0.9*|I∩ I_2|+0.8*|I^C ∩ I_1|+0.1*|I^C ∩ I_2|

        I1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        I1_and_I_size = 0
        for interval1 in I1:
            for interval2 in h_intervals:
                if interval1[1] < interval2[0] or interval2[1] < interval1[0]:  # no intersection
                    continue
                I1_and_I_size += min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])

        I_size = sum([interval[1] - interval[0] for interval in h_intervals])
        I2_and_I_size = I_size - I1_and_I_size
        I1_size = sum([interval[1] - interval[0] for interval in I1])
        I1_and_not_I_size = I1_size - I1_and_I_size
        I2_size = 1 - I1_size
        I2_and_not_I_size = I2_size - I2_and_I_size

        return 0.2 * I1_and_I_size + 0.9 * I2_and_I_size + 0.8 * I1_and_not_I_size + 0.1 * I2_and_not_I_size

    def experiment_k_range_erm_no_show (self, m, k_first, k_last, step):
        k_range = np.arange(k_first, k_last + 1, step)
        sample = self.sample_from_D(m)
        sample_X = sample[:, 0]
        sample_Y = sample[:, 1]
        arr = np.array([self.get_errors_by_k(m, k,sample_X,sample_Y) for k in k_range])
        empirical_errors=arr[:, 0]
        plt.plot(k_range, empirical_errors, label='Empirical Error')
        plt.plot(k_range, arr[:, 1], label='True Error')
        plt.legend()
        plt.xlabel("k")

        return k_range,empirical_errors


    def get_errors_by_k(self, m,k,x,y):
        interval_set, error = intervals.find_best_interval(x, y, k)
        empirical_error = error/m
        true_error = self.calc_true_error(interval_set)
        return np.array([empirical_error,true_error])

    def zo_loss (self, x,y,g):
        x_in_g = 0
        for interval in g:
            if interval[0] <= x <= interval[1]:
                x_in_g=1
                break
        if (y==0 and x_in_g==1) or (y==1 and  x_in_g==0):
            return 1
        return 0

if __name__ == '__main__':
    ass = Assignment2()
    #ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    #ass.experiment_k_range_erm(1500, 1, 10, 1)
    #ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)


