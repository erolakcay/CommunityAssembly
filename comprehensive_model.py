#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script that simulates assembly of ecological communities.

Jimmy J. Qian and Erol Akçay, 2020. "The balance of interaction types
determines the assembly and stability of ecological communities"
"""

import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import numpy as np
from scipy.stats import halfnorm
from scipy import integrate
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import sys
import warnings

class Community(object):
    """
    An ecological community whose assembly is simulated over time. The class
    is initialized by passing in four mandatory parameters and some optional parameters.
    """

    def __init__(self, C, Pc, Pm, h, plotbool=False, plot_failed_eq=False,
                 study_selection=False, relative_eq_criteria=False,
                 IIM_bool = False, variable_r = False, VUM_bool=False,
                 lamb = 0.5):
        """
        Takes in parameters that are varied across communities.
        Initialize default parameters of the community as attributes.

        Input:
            C (float): desired connectivity of the community
            Pc (float): proportion of competitive interactions
            Pm (float): proportion of mutualistic interactions
            h (float): half-saturation constant for type II functional response
            plotbool (boolean): whether or not to plot populations over time
            plot_failed_eq (boolean): whether to plot dynamics at equilibria that do not converge before the time limit
            study_selection (boolean): set True to run the analysis of selection as described for Figure 5, S14-17
            relative_eq_criteria (boolean): set True to use equilibration criteria that is relative to population sizes
            IIM_bool (boolean): set True to implement the interchangeable interactions model, otherwise use unique interactions model (make sure VUM_bool set to False)
            variable_r (boolean): set True to vary r between -1 and 1 based on the proportion of predatory interactions
            VUM_bool (boolean): set True to implement the variable uniqueness model (make sure IIM_bool is set to False)
            lamb (float): uniqueness coefficient (lambda) in the VUM
        """
        # Simulation parameters that are varied across communities. Taken as inputs
        self.C = C
        self.Pc = Pc
        self.Pm = Pm
        self.halfsat = h
        self.plotbool = plotbool
        self.plot_failed_eq = plot_failed_eq
        self.study_selection = study_selection
        self.relative_eq_criteria = relative_eq_criteria
        self.IIM_bool = IIM_bool
        self.variable_r = variable_r
        self.VUM_bool = VUM_bool
        self.lamb = lamb

        # Additional simulation parameters and variables
        self.S = 10 # number of initial species
        self.numspecies = self.S # number of total species, this changes over time
        self.numspecies_eq = [self.numspecies] # list of species richness at each equilibrium
        self.selfinteraction = -1.0 # self-regulation coefficient (diagonals of the interaction matrix)
        sigma = 0.5 # half-normal distribution scale parameter
        self.K = 100 # carrying capacity of each species
        sigma_c = sigma / self.K # half-normal distribution scale parameter for competitive interaction
        self.strengthdist = halfnorm(scale=sigma) # half-normal distribution for mutualistic and exploitative interactions
        self.strengthdist_comp = halfnorm(scale=sigma_c) # half-normal distribution for competitive interactions
        self.r = 1 # intrinsic growth rate of all species
        self.eqcriteria = 0.001 # equilibration criteria for changes in population
        self.eqcriteriamatrix = np.full((1,self.S),self.eqcriteria) # 1 by S matrix of equilibrium criteria
        self.extinctthres = 0.01 # extinction threshold for population sizes
        self.timelim = 5000 # time limit to wait for equilibrium
        self.failedinvasionlim = 1000 # failed invasion threshold
        self.numspecieslim = 750 # limit on species richness
        self.steadytime = 0 # equilibrium at which species richness converges
        self.pvalthres = 0.01 # p-value threshold to reject null hypothesis
        self.window = 1000 # number of equilibria analyzed in augmented Dickey-Fuller test
        self.poststeadywindow = 500 # number of equilibria simulated after steady state is reached
        self.t_start = 1. # simulation start time
        self.t_end = 10**9 # time limit of simulation; this is never reached
        self.currenttime=self.t_start
        self.integratedtime = self.t_start
        self.extinctiontimes = [] # list of extinction times
        self.numextinct = 0 # number/counter of total extinctions
        self.eqs = [0] # list of times that the system is at equilibrium
        self.failedinvasiontimes = [] # list of times a failed invasion occurs
        self.failedinvasionsteady=0 # number of failed invasions during the steady state
        self.extinctionsteady=0 # number of extinctions during the steady state
        self.numspeciesatsteady=0 # species richness when steady state is reached
        introtime=np.full((1,self.S), 0) # 1 by S matrix describing introduction times of species
        self.introtime = introtime.flatten()
        self.ages=[] # list of the species persistence for every species that existed
        self.wait=[] # times the time limit to wait for equilibrium is reached

        if plotbool:
            self.time = [self.t_start]
        if plot_failed_eq:
            self.failed_eq_counter = 0
        if study_selection:
            self.sampledmatrices = [] # list of sampled interaction matrices
            self.sampledinvaderrows=[] # list of interaction matrix rows of sampled successful invaders
            self.sampledinvadercols=[] # list of interaction matrix cols of sampled successful invaders
            self.sampledextinctionrows=[] # list of interaction matrix rows of sampled species that go extinct
            self.sampledextinctioncols=[] # list of interaction matrix cols of sampled species that go extinct

            # sampling scheme
            self.inv_ext_sampleinterval = 10 # sampling interval as described in Methods
            self.numcommunitysamples=100 # how many matrices we sample after steady state
            self.ext_samplecounter = 0 # counter to sample species that go exctinct with given interval
        if relative_eq_criteria:
            self.rel_eq_criteria = 1/10**6

        # create integration object
        ode = integrate.ode(self.eq_system)
        ode.set_integrator('dopri5',nsteps=1)
        warnings.filterwarnings("ignore", category=UserWarning) # ignore warnings of integrating step by step
        self.ode = ode

        # Boolean attributes reporting whether various simulation limits are reached
        self.steady = False # boolean indicating whether community has come to the steady state
        self.broken = False # boolean indicating whether community has come to un-invadable equilibrium or exceeds limit on species richness

    def start(self):
        """
        Construct the initial community, create uninitialized interaction
        matrix, initialize interaction matrix, and impose desired connectivity
        onto interaction matrix. Must call this method after initialization.
        """
        S = self.S
        # construct the initial community
        X0 = np.random.rand(S) # matrix describing initial population size of each species
        X0 = X0 * 10 # initialize populations of initial species with uniform random samples from 0 to 10
        self.R = np.full((S,1),self.r) # S by 1 matrix describing intrinsic growth rate of each species
        self.population = X0 # current population
        self.prevpop = X0 # previous population
        if self.plotbool:
            self.populations = X0 # save full population histories

        # create uninitialized interaction matrix
        A = np.zeros((S,S)) # uninitialized S by S interaction matrix, a_ij is j's effect on i
        A_c = np.zeros((S,S)) # uninitialized interaction matrix with competitive terms only
        A_m = np.zeros((S,S))  # uninitialized interaction matrix with mutualistic terms only
        A_e_pos = np.zeros((S,S))  # uninitialized interaction matrix with positive exploitative terms only
        A_e_neg = np.zeros((S,S))  # uninitialized interaction matrix with negative exploitative terms only

        # initialize interaction matrix, impose desired connectivity onto interaction matrix
        for i in range(0,S):

            if self.variable_r:
                num_interaction = 0
                num_mutualistic = 0
                num_predatory = 0

            for j in range(i+1,S): # shouldn't be range(i,S) since j and i shouldn't be equal
                if np.random.uniform(0,1) <= self.C: # then there is an interaction between i and j
                    randnum = np.random.uniform(0,1)
                    if self.variable_r:
                        num_interaction = num_interaction + 1
                    if randnum >= (1-self.Pm): # then mutualistic interaction
                        # inverse transform sampling: use percent point fxn of the half normal
                        A[i,j] = self.strengthdist.ppf(np.random.uniform(0,1)) # interaction strength
                        A[j,i] = self.strengthdist.ppf(np.random.uniform(0,1)) # not necessarily symmetric

                        A_m[i,j] = A[i,j]
                        A_m[j,i] = A[j,i]

                        if self.variable_r:
                            num_mutualistic = num_mutualistic + 1

                    elif randnum <= self.Pc: # then competitive interaction
                        A[i,j] = -1 * self.strengthdist_comp.ppf(np.random.uniform(0,1))
                        A[j,i] = -1 * self.strengthdist_comp.ppf(np.random.uniform(0,1))

                        A_c[i,j] = A[i,j]
                        A_c[j,i] = A[j,i]

                    else: # exploitative interaction
                        if np.random.uniform(0,1) <= 0.5: # i benefits, j suffers
                            A[i,j] = self.strengthdist.ppf(np.random.uniform(0,1))
                            A[j,i] = -1 * self.strengthdist.ppf(np.random.uniform(0,1))

                            A_e_pos[i,j] = A[i,j]
                            A_e_neg[j,i] = A[j,i]

                            if self.variable_r:
                                num_predatory = num_predatory + 1

                        else: # i suffers, j benefits
                            A[i,j] = -1 * self.strengthdist.ppf(np.random.uniform(0,1))
                            A[j,i] = self.strengthdist.ppf(np.random.uniform(0,1))

                            A_e_neg[i,j] = A[i,j]
                            A_e_pos[j,i] = A[j,i]

            if self.variable_r:
                if num_interaction == num_mutualistic:
                    self.R[i,0] = 1
                else:
                    self.R[i,0] = 1 - 2 * num_predatory / (num_interaction - num_mutualistic)

        np.fill_diagonal(A, self.selfinteraction ) # all self-interactions are set to the same value

        self.A = A
        self.A_c = A_c
        self.A_m = A_m
        self.A_e_pos = A_e_pos
        self.A_e_neg = A_e_neg

    def eq_system(self,t,X):
        """
        Solve the differential equation over t for initial condition X.
        X is a 1 by S array.
        Returns the 1 by S matrix of population growth.
        """
        Xnew = X.reshape((-1,1)) # make it into 2D S by 1 matrix
        if self.VUM_bool: # VUM
            X_m = Xnew*self.lamb / (self.halfsat + Xnew*self.lamb)
            m_mask = self.A_m > 0
            m_mask = m_mask.astype(float)
            m_sums = np.dot(m_mask, Xnew)
            pred_mask = self.A_e_pos > 0
            pred_mask = pred_mask.astype(float)
            pred_sums = np.dot(pred_mask, Xnew)

            output = Xnew * (self.R + self.selfinteraction * Xnew / self.K + \
                             np.dot(self.A_c, Xnew) + \
                             np.dot(self.A_m, Xnew*(1-self.lamb))/(self.halfsat+m_sums*(1-self.lamb)) + \
                             np.dot(self.A_e_pos, Xnew*(1-self.lamb))/(self.halfsat+pred_sums*(1-self.lamb)) + \
                             np.dot(self.A_e_neg, Xnew*(1-self.lamb)/(self.halfsat+pred_sums*(1-self.lamb))) +\
                             np.dot(self.A_m, X_m) + \
                             np.dot(self.A_e_pos, X_m) + \
                             np.dot(self.A_e_neg, Xnew*self.lamb)/(self.halfsat + Xnew*self.lamb))
        elif self.IIM_bool: # IIM
            m_mask = self.A_m > 0
            m_mask = m_mask.astype(float)
            m_sums = np.dot(m_mask, Xnew)
            pred_mask = self.A_e_pos > 0
            pred_mask = pred_mask.astype(float)
            pred_sums = np.dot(pred_mask, Xnew)

            output = Xnew * (self.R + self.selfinteraction * Xnew / self.K + \
                             np.dot(self.A_c, Xnew) + \
                             np.dot(self.A_m, Xnew)/(self.halfsat+m_sums) + \
                             np.dot(self.A_e_pos, Xnew)/(self.halfsat+pred_sums) + \
                             np.dot(self.A_e_neg, Xnew/(self.halfsat+pred_sums)))
        else: # UIM
            X_m = Xnew / (self.halfsat + Xnew)
            output = Xnew * (self.R + self.selfinteraction * Xnew / self.K + \
                             np.dot(self.A_c, Xnew)+ np.dot(self.A_m, X_m) + \
                             np.dot(self.A_e_pos, X_m) + \
                             np.dot(self.A_e_neg, Xnew)/(self.halfsat + Xnew))
        output = output.T # transpose back to 1 by S
        return output.flatten()

    def introduce(self, currenteq):
        """
        Add new species once equilibrium is reached and iterate until
        next equilibrium. Input is integer number of the current equilibrium.
        """

        if self.numspecies>self.numspecieslim:
            self.broken = True

        if self.broken:
            return

        self.currenttime = self.integratedtime

        if self.currenttime >= self.t_end: # do not iterate if t_end has already been reached
            return
        if self.currenttime > self.t_start: # don't add species/check for extinction if first iteration

            # check for extinction, and remove the species immediately if extinct
            boolmask = self.population > self.extinctthres

            # extract the ages of extinct species
            extinctbirth = self.introtime[np.logical_not(boolmask)]

            # remove extinct species
            self.population = self.population[boolmask]
            self.introtime = self.introtime[boolmask]
            self.R = self.R[boolmask]
            if self.plotbool:
                self.populations = self.populations[boolmask]

            # update running total of number of extinctions
            newextinct = self.numspecies - self.population.shape[0]
            self.numextinct = self.numextinct + newextinct

            if self.steady == True:
                self.extinctionsteady = self.extinctionsteady + newextinct

                if self.study_selection:
                    self.ext_samplecounter = self.ext_samplecounter + newextinct

                    # sample the species that go extinct
                    if self.ext_samplecounter >= self.inv_ext_sampleinterval:
                        A_extinct_rows = self.A[np.logical_not(boolmask)]
                        A_extinct_cols = self.A.T[np.logical_not(boolmask)]
                        self.sampledextinctionrows.append(A_extinct_rows[0])
                        self.sampledextinctioncols.append(A_extinct_cols[0])
                        self.ext_samplecounter = self.ext_samplecounter % self.inv_ext_sampleinterval

            self.numspecies = self.population.size
            self.eqcriteriamatrix = np.full((1,self.numspecies),self.eqcriteria)

            # remove the extinct species rows from the interaction matrix
            self.A = self.A[boolmask]
            self.A_c = self.A_c[boolmask]
            self.A_m = self.A_m[boolmask]
            self.A_e_pos = self.A_e_pos[boolmask]
            self.A_e_neg = self.A_e_neg[boolmask]

            # remove the extinct species cols from interaction matrix
            self.A = self.A.T[boolmask]
            self.A_c = self.A_c.T[boolmask]
            self.A_m = self.A_m.T[boolmask]
            self.A_e_pos = self.A_e_pos.T[boolmask]
            self.A_e_neg = self.A_e_neg.T[boolmask]

            # invert back after indexing the target col as a row in transpose
            self.A = self.A.T
            self.A_c = self.A_c.T
            self.A_m = self.A_m.T
            self.A_e_pos = self.A_e_pos.T
            self.A_e_neg = self.A_e_neg.T

            if newextinct > 0:
                self.extinctiontimes.append(self.ode.t)
                self.ages.extend(currenteq - extinctbirth)# save the persistence of extinct species

            invasionfailed = True
            failedcounter = 0 # number of failed invasions at this particular equilibrium

            while invasionfailed: # create a new species, and check if it will invade

                # initialize
                newrow = np.zeros((1,self.numspecies)) # new row to be added into A
                newcol = np.zeros((self.numspecies+1,1)) # new column to be added into A, extra term because hstack after newrow
                newrow_c = np.zeros((1,self.numspecies)) # new row only with incoming competitive interactions
                newrow_m = np.zeros((1,self.numspecies)) # new row only with incoming mutualistic interactions
                newrow_e_pos = np.zeros((1,self.numspecies)) # new row only with incoming positive exploitative interactions
                newrow_e_neg = np.zeros((1,self.numspecies)) # new row only with incoming negative exploitative interactions
                newcol_c = np.zeros((self.numspecies+1,1))
                newcol_m = np.zeros((self.numspecies+1,1))
                newcol_e_pos = np.zeros((self.numspecies+1,1))
                newcol_e_neg = np.zeros((self.numspecies+1,1))

                if self.variable_r:
                    num_interaction = 0
                    num_mutualistic = 0
                    num_predatory = 0

                for j in range(0,self.numspecies):
                    if np.random.uniform(0,1) <= self.C: # then there is an interaction between i and j
                        randnum = np.random.uniform(0,1)
                        if self.variable_r:
                            num_interaction = num_interaction + 1
                        if randnum >= (1-self.Pm): # then mutualistic interaction
                            newrow[0,j]= self.strengthdist.ppf(np.random.uniform(0,1))
                            newcol[j,0]= self.strengthdist.ppf(np.random.uniform(0,1))

                            newrow_m[0,j]= newrow[0,j]
                            newcol_m[j,0]= newcol[j,0]

                            if self.variable_r:
                                num_mutualistic = num_mutualistic + 1

                        elif randnum <= self.Pc: # then competitive interaction
                            newrow[0,j]= -1 * self.strengthdist_comp.ppf(np.random.uniform(0,1))
                            newcol[j,0]= -1 * self.strengthdist_comp.ppf(np.random.uniform(0,1))

                            newrow_c[0,j]= newrow[0,j]
                            newcol_c[j,0]= newcol[j,0]

                        else: # exploitative interaction
                            if np.random.uniform(0,1) <= 0.5: # i benefits, j suffers
                                newrow[0,j]= self.strengthdist.ppf(np.random.uniform(0,1))
                                newcol[j,0]= -1 * self.strengthdist.ppf(np.random.uniform(0,1))

                                newrow_e_pos[0,j]= newrow[0,j]
                                newcol_e_neg[j,0]= newcol[j,0]

                                if self.variable_r:
                                    num_predatory = num_predatory + 1

                            else: # i suffers, j benefits
                                newrow[0,j]= -1 * self.strengthdist.ppf(np.random.uniform(0,1))
                                newcol[j,0]= self.strengthdist.ppf(np.random.uniform(0,1))

                                newrow_e_neg[0,j]= newrow[0,j]
                                newcol_e_pos[j,0]= newcol[j,0]

                # Check to see if invasion can occur:
                if self.variable_r:
                    if num_interaction == num_mutualistic:
                        new_r = 1
                    else:
                        new_r =  1 - 2 * num_predatory / (num_interaction - num_mutualistic)
                else:
                    new_r = self.r
                xjcol = self.population
                if self.VUM_bool: # VUM
                    X_m = xjcol*self.lamb / (self.halfsat + xjcol*self.lamb)
                    m_mask = newrow_m > 0
                    m_mask = m_mask.astype(float)
                    m_sums = np.dot(m_mask, xjcol)
                    pred_mask = newrow_e_pos > 0
                    pred_mask = pred_mask.astype(float)
                    pred_sums = np.dot(pred_mask, xjcol)
                    exploiter_mask = self.A_e_pos > 0
                    exploiter_mask = exploiter_mask.astype(float)
                    exploiter_sums = np.dot(exploiter_mask, xjcol)
                    # invasion condition
                    if (new_r + np.dot(newrow_c, xjcol) + \
                        np.dot(newrow_m, xjcol*(1-self.lamb))/(self.halfsat + m_sums*(1-self.lamb)) + \
                        np.dot(newrow_e_pos, xjcol*(1-self.lamb))/(self.halfsat + pred_sums*(1-self.lamb)) +\
                        np.dot(newrow_e_neg, xjcol*(1-self.lamb)/(self.halfsat + exploiter_sums*(1-self.lamb))) +\
                        np.dot(newrow_m, X_m) +\
                        np.dot(newrow_e_pos, X_m) + \
                        np.dot(newrow_e_neg, xjcol)*self.lamb/self.halfsat) > 0:
                        invasionfailed = False
                    else:
                        self.failedinvasiontimes.append(self.currenttime) # track the times of failed invasions
                        failedcounter += 1
                        if self.steady == True:
                            self.failedinvasionsteady += 1
                        if failedcounter > self.failedinvasionlim:
                            self.broken = True
                            print('Uninvadable equilibrium')
                            return
                elif self.IIM_bool: # IIM
                    m_mask = newrow_m > 0
                    m_mask = m_mask.astype(float)
                    m_sums = np.dot(m_mask, xjcol)
                    pred_mask = newrow_e_pos > 0
                    pred_mask = pred_mask.astype(float)
                    pred_sums = np.dot(pred_mask, xjcol)
                    exploiter_mask = self.A_e_pos > 0
                    exploiter_mask = exploiter_mask.astype(float)
                    exploiter_sums = np.dot(exploiter_mask, xjcol)
                    # invasion condition
                    if (new_r + np.dot(newrow_c, xjcol) + \
                        np.dot(newrow_m, xjcol)/(self.halfsat + m_sums) + \
                        np.dot(newrow_e_pos, xjcol)/(self.halfsat + pred_sums) +\
                        np.dot(newrow_e_neg, xjcol/(self.halfsat + exploiter_sums))) > 0:
                        invasionfailed = False
                    else:
                        self.failedinvasiontimes.append(self.currenttime) # track the times of failed invasions
                        failedcounter += 1
                        if self.steady == True:
                            self.failedinvasionsteady += 1
                        if failedcounter > self.failedinvasionlim:
                            self.broken = True
                            print('Uninvadable equilibrium')
                            return
                else: # UIM
                    X_m = xjcol / (self.halfsat + xjcol)
                    # invasion condition
                    if (new_r + np.dot(newrow_c, xjcol) + np.dot(newrow_m, X_m) + \
                        np.dot(newrow_e_pos, X_m) + \
                        np.dot(newrow_e_neg, xjcol)/self.halfsat) > 0:
                        invasionfailed = False
                    else:
                        self.failedinvasiontimes.append(self.currenttime) # track the times of failed invasions
                        failedcounter += 1
                        if self.steady == True:
                            self.failedinvasionsteady += 1
                        if failedcounter > self.failedinvasionlim:
                            self.broken = True
                            print('Uninvadable equilibrium')
                            return

            # add new species into interaction matrix once invasion condition is met
            newcol[self.numspecies,0] = self.selfinteraction
            self.A = np.vstack((self.A,newrow))
            self.A = np.hstack((self.A,newcol))

            self.A_c = np.vstack((self.A_c,newrow_c))
            self.A_c = np.hstack((self.A_c,newcol_c))
            self.A_m = np.vstack((self.A_m,newrow_m))
            self.A_m = np.hstack((self.A_m,newcol_m))
            self.A_e_pos = np.vstack((self.A_e_pos,newrow_e_pos))
            self.A_e_pos = np.hstack((self.A_e_pos,newcol_e_pos))
            self.A_e_neg = np.vstack((self.A_e_neg,newrow_e_neg))
            self.A_e_neg = np.hstack((self.A_e_neg,newcol_e_neg))

            if (self.study_selection and self.steady == True and
                ((currenteq - self.steadytime) % self.inv_ext_sampleinterval == 0)):
                newrow = np.hstack((newrow.flatten(),self.selfinteraction))
                self.sampledinvaderrows.append(newrow)
                self.sampledinvadercols.append(newcol)

            newpop = 10 * np.random.uniform(0,1) # initial population of new species
            self.prevpop = np.append(self.population,0)
            self.population = np.append(self.population, newpop)

            if self.plotbool:
                newrowpop = np.zeros((1,self.populations.shape[1]))
                newrowpop[0,self.populations.shape[1]-1]= newpop
                self.populations = np.vstack((self.populations, newrowpop))

            self.introtime = np.append(self.introtime, currenteq)
            self.eqs.append(self.currenttime)

            # extend matrices for eqcriteria and R and extinction threshold
            self.eqcriteriamatrix=np.append(self.eqcriteriamatrix, self.eqcriteria)
            self.R = np.vstack((self.R,np.array([new_r])))
            self.numspecies += 1
            self.numspecies_eq.append(self.numspecies)
            lastpop = self.population

        else: # this is the first iteration of simulation
            lastpop = self.population

        if self.plot_failed_eq:
            self.eq_population = self.population
            self.this_eq_time = [self.currenttime]

        # integrate population dynamics until the next equilibrium
        self.ode.set_initial_value(lastpop.flatten(),self.currenttime)
        while (self.ode.t<self.t_end and (self.ode.t - self.currenttime)<self.timelim):
            if self.ode.t > self.t_start: # do not check for equilibrium/extinction on first pass
                check = np.absolute(self.population - self.prevpop)
                if self.relative_eq_criteria:
                    relative_criteria = self.population * self.rel_eq_criteria
                    relative_criteria = np.clip(relative_criteria, a_min=self.eqcriteria, a_max=None)
                    check = check < relative_criteria
                else:
                    check = check < self.eqcriteriamatrix
                if np.all(check): # if all growth rates are less than equilibrium criteria
                    return

            self.ode.integrate(self.t_end)
            self.integratedtime = self.ode.t
            self.prevpop = self.population
            self.population = self.ode.y # update population
            if (self.ode.t - self.currenttime) > self.timelim: # did not converge within time limit
                self.wait.append(currenteq)

                if self.plot_failed_eq: # plot the dynamics
                    self.plot_dynamics_single_eq()
                    self.failed_eq_counter = self.failed_eq_counter+1

            if self.plotbool:
                self.time.append(self.ode.t)
                self.populations = np.column_stack((self.populations,self.ode.y))
            if self.plot_failed_eq:
                self.this_eq_time.append(self.ode.t)
                self.eq_population = np.column_stack((self.eq_population,self.ode.y))


    def iterate(self, limit=5000):
        """
        Introduce species for the specified number of iterations.
        Input (optioinal): limit on the number of equilibria.
        Saves data as Numpy .npz file.
        To study selection (Figure 5, S14-17), use iterate_selection() instead.
        """
        self.iterationslimit=limit
        for i in range(0,limit):
            self.introduce(i)
            if self.broken:
                break

            if self.steady == False:
                if (i >= self.window - 1): # start checking for steady state
                    series=np.array(self.numspecies_eq)[(i+1-self.window):(i+1)]
                    pval = adfuller(series)[1] # calculate p-value of augmented Dickey-Fuller test
                    if pval< self.pvalthres:
                        self.steadytime = i # equilibrium number in which steady state has been reached
                        self.steady = True
                        self.numspeciesatsteady = self.numspecies
            if self.steady==True:
                if i >= self.steadytime+self.poststeadywindow: # stop the simulation
                    break

        # save the persistence of the species in the final population
        self.ages.extend(len(self.eqs) - self.introtime)

        # export the data from the simulation
        paramvals = np.array([self.C, self.Pc, self.Pm, self.halfsat])
        brokenarray = np.array([self.broken])
        steadyarray = np.array([self.steady])
        waitarray = np.array([self.wait])
        aftersteady = np.array([self.extinctionsteady, self.failedinvasionsteady, self.numspeciesatsteady])
        np.savez('output_'+str(self.C)+'_'+str(self.Pc)+'_'+str(self.Pm)+'_'+str(self.halfsat),\
                 paramvals = paramvals, population = self.population,\
                 failedinvasiontimes=self.failedinvasiontimes, extinctiontimes = self.extinctiontimes,\
                 eqs = self.eqs, numspecies_eq = self.numspecies_eq, broken=brokenarray, \
                 steadytime = self.steadytime, ages=self.ages, steady=steadyarray,\
                 aftersteady = aftersteady, waitarray = waitarray)

    def iterate_selection(self, limit=5000):
        """
        Use this method, instead of iterate(), to study selection (Figure 5, S14-17).
        Introduce species for the specified number of iterations.
        Input (optioinal): limit on the number of equilibria.
        Saves data as Numpy .npz file.
        Community must be initialized with study_selection = True.
        """
        if self.study_selection == False:
            raise RuntimeError('Must have study_selection=True to study selection.')
        self.iterationslimit=limit
        for i in range(0,limit):
            self.introduce(i)
            if self.broken:
                break

            if self.steady == False:
                if (i >= self.window - 1): # start checking for steady state
                    series=np.array(self.numspecies_eq)[(i+1-self.window):(i+1)]
                    pval = adfuller(series)[1] # calculate p-value of augmented Dickey-Fuller test
                    if pval< self.pvalthres:
                        self.steadytime = i # equilibrium number in which steady state has been reached
                        self.steady = True
                        self.numspeciesatsteady = self.numspecies
                        self.poststeadywindow = self.numcommunitysamples * self.numspeciesatsteady

            if self.steady==True:
                if i >= self.steadytime+self.poststeadywindow: # stop the simulation
                    break
                if (i - self.steadytime) % self.numspeciesatsteady == 0:
                    self.sampledmatrices.append(self.A)

        # save the persistence of the species in the final population
        self.ages.extend(len(self.eqs) - self.introtime)

        # export the data from the simulation
        paramvals = np.array([self.C, self.Pc, self.Pm, self.halfsat])
        brokenarray = np.array([self.broken])
        steadyarray = np.array([self.steady])
        waitarray = np.array([self.wait])
        aftersteady = np.array([self.extinctionsteady, self.failedinvasionsteady, self.numspeciesatsteady])
        np.savez('selection_'+str(self.C)+'_'+str(self.Pc)+'_'+str(self.Pm)+'_'+str(self.halfsat),\
                 paramvals = paramvals, population = self.population,\
                 failedinvasiontimes=self.failedinvasiontimes, extinctiontimes = self.extinctiontimes,\
                 eqs = self.eqs, numspecies_eq = self.numspecies_eq, broken=brokenarray, \
                 steadytime = self.steadytime, ages=self.ages, steady=steadyarray,\
                 aftersteady = aftersteady, waitarray = waitarray, \
                 sampledmatrices=self.sampledmatrices, sampledinvaderrows=self.sampledinvaderrows,\
                 sampledinvadercols=self.sampledinvadercols,\
                 sampledextinctionrows=self.sampledextinctionrows,\
                 sampledextinctioncols=self.sampledextinctioncols)

    def plot_richness(self):
        """
        Plot the species richness over time over entire community history.
        """
        # plot the number of species over time using eqs
        plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        eq_num = np.arange(0,len(self.eqs))
        ax.plot(eq_num, self.numspecies_eq, 'r')
        if self.steady == True:
            ax.axvline(x=self.steadytime, color = 'grey', ls='--') # plot vertical line when steady state is reached
        ax.set_ylabel('Species richness')
        ax.set_xlabel('Equilibria')
        ax.set_title(r'C = %.1f, $P_c = $ %.1f, $P_m = $%.1f' %(self.C, self.Pc, self.Pm))
        fig.savefig('species_eqnum '+str(self.iterationslimit)+'_'+str(self.C)+'_'\
                    +str(self.Pc)+'_'+str(self.Pm)+'_'+str(self.halfsat)+'.eps')

    def plot_dynamics(self):
        """
        Plot the species population dynamics over the entire community history.
        Community must be initialized with plotbool = True.
        """
        if self.plotbool == False:
            raise RuntimeError('Must have plotbool=True to plot dynamics.')

        plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # plot extant species
        for i in range(0,self.numspecies):
            ax.plot(self.time, self.populations[i])
        for xc in self.eqs: # plot vertical line whenever new species is introduced
            plt.axvline(x=xc, color = 'grey', ls=':')
        ax.set_ylabel('Population size')
        ax.set_xlabel('Time')
        ax.set_title(r'C = %.1f, $P_c = $ %.1f, $P_m = $%.1f' %(self.C, self.Pc, self.Pm))
        fig.savefig('populations '+str(self.iterationslimit)+'_'+str(self.C)+'_'\
                    +str(self.Pc)+'_'+str(self.Pm)+str(self.halfsat)+'.eps')

    def plot_dynamics_single_eq(self):
        """
        Plot the species population dynamics for a single equilibrium.
        Community must be initialized with plot_failed_eq = True.
        """
        if self.plot_failed_eq == False:
            raise RuntimeError('Must have plot_failed_eq=True to plot dynamics.')

        plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # plot extant species
        for i in range(0,self.eq_population.shape[0]):
            ax.plot(self.this_eq_time, self.eq_population[i],alpha=0.5)
        ax.set_ylabel('Population size')
        ax.set_xlabel('Time')
        ax.set_title(r'C = %.1f, $P_c = $ %.1f, $P_m = $%.1f, h = %.1f' %(self.C, self.Pc, self.Pm, self.halfsat))
        fig.savefig('failed-eq'+str(self.failed_eq_counter)+'_'+str(self.iterationslimit)+'_'+str(self.C)+'_'\
                    +str(self.Pc)+'_'+str(self.Pm)+'_'+str(self.halfsat)+'.eps')

def run_simulation(command_line):
    """
    Create combinations of parameters and simulate communities.
    Takes an index from 0 to 593 as an argument. This index determines the
    parameter values that are used to create the community.
    """
    # create lists all possible parameter values for Pc, Pm, C, h
    list_Pc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    list_Pm = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    list_C=[0.1,0.5,0.9]
    list_h=[20,100,500]

    # create array of all possible combinations, subject to normalization
    numcombo = len(list_Pc) * len(list_Pm)
    combo = list(itertools.product(list_Pc, list_Pm)) # all possible combinations
    checknorm = []
    for i in range(0,numcombo): # need to check normalization
        checknorm.append(combo[i][0] + combo[i][1] <= 1)
    combo=np.array(combo)
    checknorm=np.array(checknorm)
    combo = combo[checknorm] # mask combo so that normalization is ensured

    fullcombo = list(itertools.product(combo,list_C,list_h))
    params = fullcombo[command_line]

    Pc=params[0][0]
    Pm=params[0][1]
    C=params[1]
    h=params[2]

    simulation = Community(C, Pc, Pm, h, study_selection=False, IIM_bool = False,
                           variable_r = False, VUM_bool = False, lamb=0.5)
    simulation.start()
    #simulation.iterate_selection()
    simulation.iterate()
    #simulation.plot_richness()
    #simulation.plot_dynamics()

    return

command_line = int(sys.argv[1]) # run script with command line argument
run_simulation(command_line)
