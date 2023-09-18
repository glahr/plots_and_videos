import dit
import numpy as np
from jpype import *
from collections import deque

jarLocation = '/home/glahr/Downloads/infodynamics-dist-1.6.1/infodynamics.jar'
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

def get_init_drop_idx(data):
    my_std = deque([], maxlen=5)

    for i, di in enumerate(data):
        my_std.append(di)
        std_h = np.std(my_std)
        if std_h > 0.0001 and data[i]-data[i-1] < 0:
        # if std_h > 0.0001:
            return i


def get_pmf(data_x, data_y, n_bins):
    lims_data_x = [data_x.min(), data_x.max()]
    lims_data_y = [data_y.min(), data_y.max()]

    pxy, _, _ = np.histogram2d(data_x[0], data_y[0], bins=n_bins, range=[lims_data_x,lims_data_y])

    for p_o, p_h in zip(data_x[1:], data_y[1:]):
        pxy += np.histogram2d(p_o, p_h, bins=n_bins, range=[lims_data_x,lims_data_y])[0]
    
    pxy /= np.sum(pxy)
    pxy[pxy == 0.] = 1e-7

    dxy = dit.Distribution.from_ndarray(pxy)
    
    return dxy


def calc_te(sourceArray=[], destArray=[], k_hist=1, k_tau=1, l_hist=1, l_tau=1, delay=0):
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.initialise()
    teCalc.setProperty("k_HISTORY", str(k_hist))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l_hist))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))

    teCalc.startAddObservations()
    for s, d in zip(sourceArray, destArray):
        teCalc.addObservations(JArray(JDouble, 1)(s), JArray(JDouble, 1)(d))
    teCalc.finaliseAddObservations()
    
    result = teCalc.computeAverageLocalOfObservations()
    return result


def calc_te_and_local_te(sourceArray=[], destArray=[], k_hist=1, k_tau=1, l_hist=1, l_tau=1, delay=0):
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.initialise()
    teCalc.setProperty("k_HISTORY", str(k_hist))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l_hist))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))

    teCalc.startAddObservations()
    for s, d in zip(sourceArray, destArray):
        teCalc.addObservations(JArray(JDouble, 1)(s), JArray(JDouble, 1)(d))
    teCalc.finaliseAddObservations()
    
    result = teCalc.computeAverageLocalOfObservations()
    result_local = teCalc.computeLocalOfPreviousObservations()
    return result, result_local


def calc_local_te(sourceArray=[], destArray=[], k_hist=1, k_tau=1, l_hist=1, l_tau=1, delay=0):
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.initialise()
    teCalc.setProperty("k_HISTORY", str(k_hist))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l_hist))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))

    teCalc.startAddObservations()
    for s, d in zip(sourceArray, destArray):
        teCalc.addObservations(JArray(JDouble, 1)(s), JArray(JDouble, 1)(d))
    teCalc.finaliseAddObservations()
    
    result = teCalc.computeLocalOfPreviousObservations()
    return result


def calc_ais(sourceArray=[], k_hist=1, k_tau=1):
    aisCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
    aisCalc = aisCalcClass()
    aisCalc.setProperty("k_HISTORY", str(k_hist))
    aisCalc.setProperty("k_TAU",     str(k_tau))
    aisCalc.initialise()
    aisCalc.startAddObservations()
    for s in sourceArray:
        aisCalc.addObservations(JArray(JDouble, 1)(s))
    aisCalc.finaliseAddObservations()
    
    result = aisCalc.computeAverageLocalOfObservations()
    return result


def calc_cte(sourceArray=[], destArray=[], condArray=[], k_hist=1, k_tau=1, l_hist=1, l_tau=1, delay=0, c_hist=1, c_tau=1, c_delay=0):
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov
    cteCalc = teCalcClass()
    cteCalc.setProperty("k_HISTORY", str(k_hist))
    cteCalc.setProperty("k_TAU",     str(k_tau))
    cteCalc.setProperty("l_HISTORY", str(l_hist))
    cteCalc.setProperty("l_TAU",     str(l_tau))
    cteCalc.setProperty("DELAY",     str(delay))
    cteCalc.setProperty("COND_EMBED_LENGTHS", str(c_hist))
    cteCalc.setProperty("COND_TAUS",          str(c_tau))
    cteCalc.setProperty("COND_DELAYS",        str(c_delay))
    cteCalc.initialise()
    cteCalc.startAddObservations()
    for s, d, c in zip(sourceArray, destArray, condArray):
        cteCalc.addObservations(JArray(JDouble, 1)(s), JArray(JDouble, 1)(d), JArray(JDouble, 1)(c))
    cteCalc.finaliseAddObservations()
    
    result = cteCalc.computeAverageLocalOfObservations()
    return result