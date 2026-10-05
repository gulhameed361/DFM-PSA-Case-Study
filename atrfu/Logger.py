#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy as np

from helper import packXYZ

class IterLog:

    def __init__(self, iteration,xk,yk,zk,print_vars):
        self.iteration = iteration
        self.xk = xk
        self.yk = yk
        self.zk = zk
        self.print_vars = print_vars
        self.thetak = None
        self.objk = None
        self.chik = None
        self.IT = None
        self.trustRadius = None
        self.sampleRadius = None
        self.stepNorm = None
        self.fStep, self.thetaStep, self.relaxthetaStep, self.rejected, self.restoration = [False]*5


    def setRelatedValue(self,thetak=None,objk=None,chik=None,IT=None,trustRadius=None,sampleRadius=None,stepNorm=None):
        if thetak is not None:
            self.thetak = thetak
        if objk is not None:
            self.objk = objk
        if chik is not None:
            self.chik = chik
        if IT is not None:
            self.IT = IT
        if trustRadius is not None:
            self.trustRadius = trustRadius
        if sampleRadius is not None:
            self.sampleRadius = sampleRadius
        if stepNorm is not None:
            self.stepNorm = stepNorm


    def fprint(self):
        print("\n**************************************")
        print("Iteration %d:" % self.iteration)
        if self.print_vars:
            print(packXYZ(self.xk, self.yk, self.zk))
        print("thetak = %s" % self.thetak)
        print("objk = %s" % self.objk)
        print("trustRadius = %s" % self.trustRadius)
        print("sampleRadius = %s" % self.sampleRadius)
        print("stepNorm = %s" % self.stepNorm)
        print("chi = %s" % self.chik)
        print("Iteration time (s) = %s" % self.IT)
        if self.fStep:
            print("f-type step")
        if self.thetaStep:
            print("theta-type step")
        if self.relaxthetaStep:
            print("relax-theta-type step")
        if self.rejected:
            print("step rejected")
        if self.restoration:
            print("RESTORATION")

        print("**************************************\n")


class Logger:
    def __init__(self):
        # Instance attribute (a class attribute here would be SHARED across
        # Logger instances, cross-contaminating consecutive solver runs in
        # the same process, e.g. benchmark batches).
        self.iters = []

    def newIter(self,iteration,xk,yk,zk,thetak,objk,chik,IT,print_vars):
        self.iterlog = IterLog(iteration,xk,yk,zk,print_vars)
        self.iterlog.setRelatedValue(thetak=thetak,objk=objk,chik=chik,IT=IT)
        self.iters.append(self.iterlog)
    def setCurIter(self,trustRadius=None,sampleRadius=None,stepNorm=None):
        self.iterlog.setRelatedValue(trustRadius=trustRadius,sampleRadius=sampleRadius,stepNorm=stepNorm)

    def printSummary(self):
        total_time = sum(x.IT for x in self.iters if x.IT is not None)
        total_iterations = max(x.iteration for x in self.iters if x.iteration is not None)
        print("\n========== Optimization Summary ==========")
        print(f"Total Time (s): {total_time:.6f}")
        print(f"Total Iterations: {total_iterations}")
        print("==========================================\n")

    def printIteration(self,iteration):
        if(iteration<len(self.iters)):
            self.iters[iteration].fprint()
    def printVectors(self):
        for x in self.iters:
            dis = np.linalg.norm(packXYZ(x.xk-self.iterlog.xk,x.yk-self.iterlog.yk,x.zk-self.iterlog.zk),np.inf)
            print(str(x.iteration)+"\t"+str(x.thetak)+"\t"+str(x.objk)+"\t"+str(x.chik)+"\t"+"\t"+str(x.IT)+"\t"+str(x.trustRadius)+"\t"+str(x.sampleRadius)+"\t"+str(x.stepNorm)+"\t"+str(dis))
        self.printSummary()