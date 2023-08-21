import sys, os

def blockPrint():
    '''
    Disable printout
    '''
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    '''
    Restore printout
    '''
    sys.stdout = sys.__stdout__
