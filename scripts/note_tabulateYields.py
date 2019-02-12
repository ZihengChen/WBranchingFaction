
#!/usr/bin/env python

def printYeilds(selection, nbjet='==1'):

    df = DFCutter(selection,nbjet,"mctt").getDataFrame()
    n,nVar = np.sum(df.eventWeight),np.sum(df.eventWeight**2)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    df = DFCutter(selection,nbjet,"mct").getDataFrame()
    n,nVar = np.sum(df.eventWeight),np.sum(df.eventWeight**2)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    df = DFCutter(selection,nbjet,"mcw").getDataFrame()
    n,nVar = np.sum(df.eventWeight),np.sum(df.eventWeight**2)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    df = DFCutter(selection,nbjet,"mcz").getDataFrame()
    n,nVar = np.sum(df.eventWeight),np.sum(df.eventWeight**2)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    df = DFCutter(selection,nbjet,"mcdiboson").getDataFrame()
    n,nVar = np.sum(df.eventWeight),np.sum(df.eventWeight**2)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    n, nVar = ct.getNFake(selection,nbjet)
    print("{:8.1f} +/- {:4.1f}".format(n,nVar**0.5) )
    total += n
    totalVar += nVar

    print("{:8.1f} +/- {:4.1f}".format(total,totalVar**0.5) )
    df = DFCutter(selection,ct.nbjet,"data2016").getDataFrame()
    print("{:8.1f} +/- {:4.1f}".format(np.sum(df.eventWeight),np.sum(df.eventWeight**2)**0.5) )