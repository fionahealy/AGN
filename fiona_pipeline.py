import numpy
import math
from matplotlib import pyplot
from scipy import constants
import pyfits
from astropy.io import fits
import os
# modules imported from casa - uncomment if running outside the casa prompt!
# from casa import table as tb


# FUNCTIONS
# --------------------------------------------------------------------------
def fitsconverter(loc,stem,freq):
    imhead = fits.open(loc+"/"+stem+'Q'+str(freq)+'.FITS')
    Freq = imhead[0].header['CRVAL3']
    Npix = imhead[0].header['NAXIS1']
    Q_im = fits.getdata(loc+"/"+stem+'Q'+str(freq)+'.FITS')[0,0,:,:]
    U_im = fits.getdata(loc+"/"+stem+'U'+str(freq)+'.FITS')[0,0,:,:]

    return Freq,Npix,Q_im,U_im

def FITS_to_MS(stokes,freq,loc,dest):
    input = loc+"/"+stem+stokes+str(freq)+'.FITS'
    output = dest+"/"+stem+stokes+str(freq)+'.CLEAN.image'

    importfits(fitsimage = input,imagename = output,overwrite=True)

# def MS_to_array(image,dim):
#     # potentially don't need 'dim' param as can get from data?
#     map=numpy.zeros([dim,dim])
#
#     tb.open(image,nomodify=False)
#     data=tb.getcol("map")
#     map[:,:]=data[:,:,0,0,0]
#     tb.close()
#
#     return(map)

def display_array(array,outname):
    pyplot.imshow(array)
    pyplot.savefig(outname+'.png')

def Find_RMS(Map,Box):
    RMS=1.8*numpy.std(Map[(Box[0]):(Box[2]),(Box[1]):(Box[3])])
    return(RMS)

def propagate_PANG_error(Q,U,SigmaQ,SigmaU):
    # this is taken directly from Eoin/Seb's code, using a standard
    # error propagation technique
    # nb check if numpy has a function that can do this automatically!
    Part1=1/(4*numpy.add(numpy.multiply(Q,Q),numpy.multiply(U,U)))
    Part2=numpy.sqrt(numpy.add(numpy.multiply(numpy.multiply(Q,Q),SigmaU*SigmaU),numpy.multiply(numpy.multiply(U,U),SigmaQ*SigmaQ)))

    Error=numpy.multiply(Part1,Part2)*180/math.pi
    Weight=1/(numpy.multiply(Error,Error))

    return(Error,Weight)

def RM_Calc(X="",Y="",W="",S="",No=""):
    aW=numpy.sum(W)
    aWxy=numpy.sum(numpy.multiply(numpy.multiply(X,Y),W))
    aWx=numpy.sum(numpy.multiply(X,W))
    aWy=numpy.sum(numpy.multiply(Y,W))
    aWxx=numpy.sum(numpy.multiply(numpy.multiply(X,X),W))

    eRM=0
    SigmaRM=0
    Chi2=0

    D=aW*aWxx-aWx*aWx
    RM=(aW*aWxy-aWx*aWy)/D
    Int=(aWy*aWxx-aWxy*aWx)/D

    for i in range(No):
        Chi2=Chi2+(Y[i]-(RM*X[i]+Int))*(Y[i]-(RM*X[i]+Int))
        eRM=eRM+numpy.sqrt(S[i]*S[i]*((aW*W[i]*X[i]-W[i]*aWx)/D)*((aW*W[i]*X[i]-W[i]*aWx)/D))

    return(RM,eRM,Chi2)

def Mask_Map(RM="",eRM="",Max="",dimX="",dimY="",MaxRM=""):
    for i in range(dimX):
        for j in range(dimY):

            if abs(eRM[i,j])>Max or abs(RM[i,j])>MaxRM:
                RM[i,j]=numpy.nan
                eRM[i,j]=numpy.nan

    return (RM,eRM)

def plot_map(map,outname,dest):
    map_rot = numpy.flipud(map)
    ax = pyplot.subplot()
    im = ax.imshow(map_rot,cmap='cubehelix')
    pyplot.colorbar(im)
    pyplot.savefig(dest+'/'+outname+'.png')
    pyplot.close()
# --------------------------------------------------------------------------


# INITIALIZATION
# --------------------------------------------------------------------------
stem = '0735+178_'
frequencies = 4
Shift = [1,1,1,1]
rms_box = [60,100,95,200]
subset = [300,400,624,624]
# [blcx,blcy,trcx,trcy]

# initialize empty arrays
Qs = []
Us = []
Q_rmss = []
U_rmss = []
PANGS = []
Errors = []
Sigmas = []
wavelengths = []

txtfile_dest = 'RM_SCRIPT_OUTPUT/ARRAYS'
fits_loc = 'RM_SCRIPT_OUTPUT/FITSFILES'
casa_dest = 'RM_SCRIPT_OUTPUT/CASAFILES'
png_dest = 'RM_SCRIPT_OUTPUT/PNGFILES'
# --------------------------------------------------------------------------
# PROCEDURE
# --------------------------------------------------------------------------

# loop through each frequency to set up arrays needed to make RM map
for i in range(1,frequencies+1):


    # # convert Q,U fitsfiles to CASA measurement sets
    # FITS_to_MS('Q',i,fits_loc,casa_dest)
    # FITS_to_MS('U',i,fits_loc,casa_dest)
    #
    # # provide filenames based on image stem
    # filename_Q = casa_dest+"/"+stem+'Q'+str(i)+'.CLEAN.image'
    # filename_U = casa_dest+"/"+stem+'U'+str(i)+'.CLEAN.image'
    #
    # # get info from image headers
    # Freq = (imhead(imagename=filename_Q, mode='get',hdkey='crval3'))['value']
    # Npix = (imhead(imagename=filename_Q,mode='get',hdkey='shape'))[1]
    # Wavel = numpy.divide(constants.c,Freq)
    #
    # # convert Q and U maps to numpy arrays
    # Q_array = MS_to_array(filename_Q,Npix)
    # U_array = MS_to_array(filename_U,Npix)

    Freq,Npix,Q_array,U_array=fitsconverter(fits_loc,stem,i)
    Wavel = numpy.divide(constants.c,Freq)

    # finds rms in Q and U maps using the RMS box,
    # use to calculate errors and sigmas for each freq
    Q_rms = Find_RMS(Q_array,rms_box)
    U_rms = Find_RMS(U_array,rms_box)

    # trim arrays for efficiency
    Q_array_trim = Q_array[subset[0]:subset[2],subset[1]:subset[3]]
    U_array_trim = U_array[subset[0]:subset[2],subset[1]:subset[3]]

    # MAKE PANG MAP FROM Q AND U ARRAYS
    PANG_map = numpy.multiply(0.5*numpy.arctan2(U_array_trim,Q_array_trim),180/math.pi)
    PANG_Error, PANG_Sigma = propagate_PANG_error(Q_array_trim,U_array_trim,Q_rms,U_rms)

    wavelengths.append(Wavel*Wavel)
    Qs.append(Q_array_trim)
    Us.append(U_array_trim)
    Q_rmss.append(Q_rms)
    U_rmss.append(U_rms)
    PANGS.append(PANG_map)
    Errors.append(PANG_Error)
    Sigmas.append(PANG_Sigma)

    # END LOOP

# --------------------------------------------------------------

# convert lists of 2d arrays into 3d arrays
PANGS_arr = numpy.stack(PANGS)
Errors_arr = numpy.stack(Errors)
Sigma_arr = numpy.stack(Sigmas)

# initialize RM arrays
RM_map = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))
eRM_map = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))


# fix!!!

# populate RM arrays
for i in range(0,subset[2]-subset[0]):
    for j in range(0,subset[3]-subset[1]):

        # get single pixel values
        PANG_pix = PANGS_arr[:,i,j]
        Errors_pix = Errors_arr[:,i,j]
        Sigma_pix = Sigma_arr[:,i,j]

        # initialize shifted values
        PANG_pix_shifted=numpy.zeros(4)
        CHI_pix_shifted=10000000000
        RM_shifted=666666
        eRM_shifted=0
        SigmaRM_shifted=0


        for a in range(-Shift[0],Shift[0]+1):
            PANG_pix_shifted[0]=PANG_pix[0]+a*180

            for b in range(-Shift[1],Shift[1]+1):
                PANG_pix_shifted[1]=PANG_pix[1]+b*180

                for c in range(-Shift[2],Shift[2]+1):
                    PANG_pix_shifted[2]=PANG_pix[2]+c*180

                    for d in range(-Shift[3],Shift[3]+1):
                        PANG_pix_shifted[3]=PANG_pix[3]+d*180

                        RM_pix,eRM_pix,CHI_pix = RM_Calc(wavelengths,PANG_pix_shifted,Sigma_pix,Errors_pix,4)

                        if CHI_pix<CHI_pix_shifted:
                            RM_shifted=RM_pix
                            eRM_shifted=eRM_pix
                            CHI_pix_shifted=CHI_pix

        RM_shifted = float(RM_shifted)*math.pi/180
        eRM_shifted = float(eRM_shifted)*math.pi/180

        RM_map[i,j]=RM_shifted
        eRM_map[i,j]=eRM_shifted

RM_map,eRM_map = Mask_Map(RM_map,eRM_map,10,subset[2]-subset[0],subset[3]-subset[1],50)

numpy.savetxt(txtfile_dest+"/"+'QARRF_test.txt',Qs[0])
numpy.savetxt(txtfile_dest+"/"+'UARRF_test.txt',Us[0])
numpy.savetxt(txtfile_dest+"/"+'PANGF_test.txt',PANGS_arr[0])
numpy.savetxt(txtfile_dest+"/"+'ERRORF_test.txt',Errors_arr[0])
numpy.savetxt(txtfile_dest+"/"+'SIGMAF_test.txt',Sigma_arr[0])
numpy.savetxt(txtfile_dest+"/"+'RMF_test.txt',RM_map)
numpy.savetxt(txtfile_dest+"/"+'eRMF_test.txt',eRM_map)


plot_map(RM_map,'RMmap',png_dest)
plot_map(eRM_map,'eRMmap',png_dest)

fig = pyplot.figure()

for i in range(0,len(wavelengths)):

    # print(Freq,Npix)
    # print(Q_rms,U_rms)

    ax_q = fig.add_subplot(3,4,i+1)
    Q_arr = numpy.flipud(Qs[i])
    ax_q.imshow(Q_arr)
    ax_q.plot([rms_box[0],rms_box[2],rms_box[2],rms_box[0],rms_box[0]],\
            [rms_box[1],rms_box[1],rms_box[3],rms_box[3],rms_box[1]])

    ax_u = fig.add_subplot(3,4,i+5)
    U_arr = numpy.flipud(Us[i],2)
    ax_u.imshow(U_arr)
    ax_u.plot([rms_box[0],rms_box[2],rms_box[2],rms_box[0],rms_box[0]],\
            [rms_box[1],rms_box[1],rms_box[3],rms_box[3],rms_box[1]])

    ax_pang = fig.add_subplot(3,4,i+9)
    pang_arr = numpy.flipud(PANGS_arr[i])
    ax_pang.imshow(pang_arr)


fig.savefig(png_dest+"/"+'test4.png')

os.system('rm *.log')
os.system('rm *.last')
