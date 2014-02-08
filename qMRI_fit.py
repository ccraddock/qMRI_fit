#!/usr/bin/env python

import nibabel as nb
import sys
import os
import getopt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def usage( prog_name ):

    usage = """\nUsage: %s -i <image> -t <parameter csv> {-T1|-T2} -m <mask> -o <outfile>

        image: 4D nifti file containing the image data (required)
        parameter csv: csv file containing either the TI or TE values for the
                       imaging data on a single line, seperated by commas.
                       E.G. 23,250,800,1800,3740. (required)
        -T1 or -T2: indicate whether T1 or T2 quantifiaction is to desired.
                    this determine the equations that are fit to the data 
                    (required)
        mask: a binary mask to limit the image voxels for which the  
              calculations are performed. non-zero voxels are assumed to 
              indicate voxels used in calculations (required)
        outfile: The name of the output file (required)\n\n""" \
              % (prog_name)

    sys.stderr.write(usage);

def t2Func(TE, a, T2):
    return a * np.exp( -1*np.array(TE) / T2)

def t1Func(TI, a, b, T1):
    return a - b * np.exp( -1*np.array(TI) / T1)

def main(argv=None):

    if argv is None:
        argv = sys.argv

    try:
        opts, args = getopt.getopt(argv[1:], "i:t:T:m:o:")
    except getopt.GetoptError as err:
        # print help information and exit:
        sys.stderr.write(str(err)) # will print something like "option -a not recognized"
        usage(os.path.basename(argv[0]))
        return 2

    do_debug = False
    imgFile = None
    prmFile = None
    T1mapping = False
    T2mapping = False
    mskFile = None

    for o,a in opts:
        if o == "-i":
            imgFile=a
        elif o == "-t":
            prmFile=a
        elif o == "-T":
            if a == "1":
                T1mapping=True
            elif a == "2":
                T2mapping=True
            else:
                sys.stderr.write("Error invalid mapping specification -T%s\n"\
                    %a)
                usage(os.path.basename(argv[0]))
                return 2
        elif o == "-m":
            mskFile=a
        elif o == "-o":
            outFile=a
        else:
            sys.stderr.write("Ignoring unknown argument(s) %s\n"%o)

    if ((prmFile == None) or (imgFile == None) or (mskFile == None) or \
        (outFile == None) or (T1mapping is False and T2mapping is False)):
        if prmFile == None:
            sys.stderr.write("Error: missing parameter file\n")
        if imgFile == None:
            sys.stderr.write("Error: missing image file\n")
        if (T1mapping == False and T2mapping == False):
            sys.stderr.write("Error: missing one of -T1 or -T2\n")
        if mskFile == None:
            sys.stderr.write("Error: missing mask file\n")
        if outFile == None:
            sys.stderr.write("Error: missing output file\n")
        usage(os.path.basename(argv[0]))
        return 2

    if len(args) > 0:
        sys.stderr.write("Ignoring unknown argument(s) %s\n"%args)

    # read in the image data
    try:
        img=nb.load(imgFile)
        imgData=img.get_data()
        print "Read in image data %s"%\
            (','.join([str(d) for d in imgData.shape]))
    except (IOError,nb.spatialimages.ImageFileError) as err:
        sys.stderr.write("Error: could not load %s\n"%imgFile)
        sys.stderr.write("%s\n"%str(err))
        return 2

    # read in the mask data
    try:
        msk=nb.load(mskFile)
        mskData=msk.get_data()
        print "Read in mask data %s"%\
            (','.join([str(d) for d in mskData.shape]))
    except (IOError,nb.spatialimages.ImageFileError) as err:
        sys.stderr.write("Error: could not load %s\n"%mskFile)
        sys.stderr.write("%s\n"%str(err))
        return 2

    # verify that the image dimensions of the mapping data and the mask
    # are consistent
    if imgData.shape[0:3] != mskData.shape:
        err="""Error: the image dimensions in %s (%s) do not\n \
            match the image dimension of the mask %s (%s)\n"""%(imgFile, \
            ",".join([str(d) for d in imgData.shape[0:3]]),\
            mskFile,",".join([str(d) for d in mskData.shape]))
        sys.stderr.write("%s\n"%str(err))
        return 2

    # identify the non-zero values of the mask file
    mskNdx = np.nonzero(mskData.flatten())[0]

    # reshape data to make it easier to deal with data of various dimensions,
    # this might take some time
    imgData = np.reshape(imgData, (np.prod(mskData.shape),imgData.shape[3]))
    print(imgData.shape)

    # read in the parameter data
    try:
        with open(prmFile,'r') as prmF:
            prmData=prmF.read()
    except EnvironmentError as err:
            sys.stderr.write("Error: could not read %s\n" %prmFile)
            sys.stderr.write("%s\n"%str(err))
            return 2

    # convert the parameters to a list of floats
    prms=[float(p) for p in prmData.rstrip().split(',')]
    print "Read in parameters %s"%(';'.join([str(f) for f in prms]))

    # make sure that the number of images that we have matches the number
    # of parameters, we expect that the 4th dimension should be the 'time'
    # dimension even when the images are 2D
    if len(prms) != imgData.shape[1]:
        err="""Error: number of images in %s (%d) does not\n \
            match the number of parameters in %s (%d)\n"""%(imgFile, \
            imgData.shape[1],prmFile,len(prms) )
        sys.stderr.write("%s\n"%str(err))
        return 2

    # create an image for the outputs, 3D image for each parameter
    if T1mapping == True:
        outDataSzFull = mskData.shape+(3,)
        outDataSz = (np.prod(mskData.shape),3)
    elif T2mapping == True:
        outDataSzFull = mskData.shape+(2,)
        outDataSz = (np.prod(mskData.shape),2)
    else:
        sys.stderr.write("F)r some reason neither T1 or T2 mapping is specified\n")
        return 2

    print("the size of out data is %s\n"%(','.join([str(d) for d in outDataSz])))
    outData = np.zeros(outDataSz,dtype='float')

    print "Estimating %s for %d out of %d voxels\n"%\
        ('T1' if T1mapping == True else 'T2',len(mskNdx),imgData.shape[0])

    # now we loop over the voxels, and perform the estimation 
    #ndx = mskNdx[0]
    #if ndx > 0:
    count = 0
    for ndx in mskNdx:
        if count % 1000 == 0:
            sys.stdout.write("[%d of %d]\n"%(count, len(mskNdx)))
        if T1mapping == True:
            best_p=None
            best_err=1e9
            shift_mul=np.array([1,1,1,1,1])
            for i in range(0,6):
                if i > 0:
                    shift_mul[i-1]=-1
                n_data=shift_mul*imgData[ndx,:];
                p0=[max(n_data), 2*max(n_data), 1000];
                try:
                    popt, pcov = curve_fit(t1Func, prms, n_data,p0=p0)
                except RuntimeError as err:
                    if do_debug is True:
                        sys.stderr.write("Curve Fit Error: %s\n"%(str(err)))
                    continue

                yn=t1Func(prms,popt[0],popt[1],popt[2])
                fit_err=np.sum((yn-n_data)**2)
                if fit_err < best_err:
                    best_p = popt
                    best_err = fit_err
                if do_debug is True:
                    print popt,fit_err
                    plt.plot(prms,n_data,'k-',prms,yn,'r-')
                    plt.title("error is %f"%(fit_err))
                    plt.show()
            outData[ndx,]=best_p
            if do_debug is True:
                print best_p, best_err

        elif T2mapping is True:
            popt, pcov = curve_fit(t2Func, prms, imgData[ndx,:])
            outData[ndx,]=popt

        count=count+1

    # write out the results
    try:
        outData=np.reshape(outData,outDataSzFull)
        outImg=nb.Nifti1Image(outData,img.get_affine())
        nb.save(outImg,outFile)
    except (IOError,nb.spatialimages.ImageFileError) as err:
        sys.stderr.write("Error writing output file %s\n"%(outFile))
        sys.stderr.write("%s\n"%str(err))
        return 2

    return 0

if __name__ == "__main__":
    sys.exit(main())


