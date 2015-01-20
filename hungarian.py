#!/usr/bin/env python3
# -*- utf-8 -*-
"""
An implementation of Hungarian algorithm for solving the assigment problem.
An instance of the assigment problem consists of a number of workers along
with a number of jobs and a cost matrix which gives the cost of assigning the
ith worker to the jth job at position(i, j). The goal is to find an assignment
of workers to the jobs so that no job is assigned more than one worker and so
that no worker is assigned to more than one job in such a manner so as to
minimize the total cost of completing the jobs.

In this implementation, we use workers and jobs to stand for the two vertices
sets in the bipartite graph and deal with the minimum-cost matching problem. It
is easy to transform the minimum problem to the maximum one, just by setting
cost(x, y) = M - cost(x, y), where M = max cost(x, y).
"""


import numpy as np
from numba import jit


class Hungarian:

    def __init__(self):
        pass

    def execute(self, cost_matrix):
        self.cost = np.copy(cost_matrix)
        self.N = cost_matrix.shape[0]

        # labels for jobs and workers
        self.job_labels = np.empty(self.N, dtype='float64')
        self.worker_labels = np.empty(self.N, dtype='float64')

        # matched_workers[w] - job that is matched with w
        self.matched_workers = np.empty(self.N, dtype='int32')
        # matched_jobs[j] - worker that is matched with j
        self.matched_jobs = np.empty(self.N, dtype='int32')

        # determine whether the worker is on the alternating tree(in S)
        self.committed_workers = np.empty(self.N, dtype='bool')
        # determine whether the job is on the alternating tree(in T)
        self.committed_jobs = np.empty(self.N, dtype='bool')
        # parent_worker[j] that the worker which is the parent
        # node of job j on the alternating tree
        self.parent_worker = np.empty(self.N, dtype='int32')

        # use to trace the min slack values
        self.slack = np.empty(self.N, dtype='float64')
        self.slack_worker = np.empty(self.N, dtype='int32')

        self.reduce()
        self.initialize_labels()
        self.greedy_match()

        worker = self.fetch_free_worker()
        while worker < self.N:
            self.initialize_phase(worker)
            self.execute_phase()
            worker = self.fetch_free_worker()

        path = np.array([(w, j) for w, j in enumerate(self.matched_workers)])
        total_cost = cost_matrix[path[:, 0], path[:, 1]].sum()

        return total_cost, path

    def reduce(self):
        """
        Compute the reduced cost matrix.

        Reduce the cost matrix by subtracting the smallest element of each row
        from all elements of the row as well as the smallest element of each
        column from all elements of the column.
        """
        self.cost -= self.cost.min(axis=1, keepdims=True)
        self.cost -= self.cost.min(axis=0, keepdims=True)

    def greedy_match(self):
        """
        Find a vaild matching by greedily selecting among zero cost matchings.
        """
        self.matched_workers.fill(-1)
        self.matched_jobs.fill(-1)
        greedy_match(self.cost, self.matched_workers, self.matched_jobs,
                     self.worker_labels, self.job_labels)

    def initialize_labels(self):
        """
        Initialize the labels of workers and jobs

        by assigning zeros to the jobs and assigning to each worker a label
        equal to the minimum cost among its adjacent edges.
        """
        self.job_labels.fill(0)
        self.worker_labels = self.cost.min(axis=1)

    def fetch_free_worker(self):
        """
        Return first free worker or N if all workers have been assigned a job.
        """
        for w in range(self.N):
            if self.matched_workers[w] == -1:
                return w
        else:
            return w + 1

    def initialize_phase(self, worker):
        """
        Initialization before the new phase of the algorithm.

        1. Clear the committed workers(marks for workers in S) and
           committed jobs(marks for jobs in T).
        2. Initialize the slack to the values corresponding to the given worker.
        """
        self.committed_workers.fill(False)
        self.committed_jobs.fill(False)
        self.parent_worker.fill(-1)

        self.committed_workers[worker] = True
        # cost(w, j) - l(w) - l(j) where w in S and j not in T
        self.slack = (self.cost[worker]
                      - self.worker_labels[worker] - self.job_labels)
        # slack_worker[j] - worker that cost(slack_worker[j], j) -
        # l(slack_worker[j]) - l(j) = slack[j]
        self.slack_worker.fill(worker)

    def update_labels(self, delta):
        """
        Update the labels with the given delta value

        by adding it for all committed workers and by subtracting it for all
        committed jobs. Besides, update the slack values appropriately.
        """
        N = self.N
        for w in range(N):
            if self.committed_workers[w]:
                # increase the label of worker which is comimtted(in S)
                self.worker_labels[w] += delta

        for j in range(N):
            if self.committed_jobs[j]:
                # decrease the label of job which is committed(in T)
                self.job_labels[j] -= delta
            else:
                # decrease the slack of free job whose
                # corresponding worker's label which has been increased
                self.slack[j] -= delta

    def execute_phase(self):
        """
        Do one phase of the algorithm.
        """
        cost = self.cost
        worker_labels = self.worker_labels
        job_labels = self.job_labels
        matched_workers = self.matched_workers
        matched_jobs = self.matched_jobs
        committed_workers = self.committed_workers
        committed_jobs = self.committed_jobs
        slack = self.slack
        slack_worker = self.slack_worker
        parent_worker = self.parent_worker
        N = self.N
        while True:
            # try to find N(S)
            minslack = np.inf
            minslack_job = -1
            for j in range(N):
                if not committed_jobs[j] and slack[j] < minslack:
                    minslack_job = j
                    minslack = slack[j]
            minslack_worker = slack_worker[minslack_job]

            if minslack > 0:
                # N(S) = T,
                # update labels to force N(S) != T
                self.update_labels(minslack)

            # now N(S) != T and minslack_job in N(S) - T
            parent_worker[minslack_job] = minslack_worker
            if matched_jobs[minslack_job] == -1:
                # minslack_job is free.
                # augmenting path found!
                leaf = minslack_job
                while leaf != -1:
                    parent = parent_worker[leaf]
                    grandparent = matched_workers[parent]
                    matched_workers[parent] = leaf
                    matched_jobs[leaf] = parent
                    leaf = grandparent
                break
            else:
                # minslack_job is not free.
                # add minslack_job to T, add the worker matched with
                # minslack_job to S
                committed_jobs[minslack_job] = True
                worker = matched_jobs[minslack_job]
                committed_workers[worker] = True

                # update slack since we add a new worker to S
                for j in range(self.N):
                    if not committed_jobs[j]:
                        new_slack = (cost[worker, j]
                                     - worker_labels[worker] - job_labels[j])
                        if new_slack < slack[j]:
                            slack[j] = new_slack
                            slack_worker[j] = worker


@jit
def greedy_match(cost_matrix, matched_workers, matched_jobs,
                 worker_labels, job_labels):
    """
    Use numba to speed up.
    """
    N = cost_matrix.shape[0]

    for w in range(N):
        for j in range(N):
            if (matched_workers[w] == -1 and matched_jobs[j] == -1
                and cost_matrix[w, j] - worker_labels[w] - job_labels[j] == 0):
                matched_workers[w] = j
                matched_jobs[j] = w
