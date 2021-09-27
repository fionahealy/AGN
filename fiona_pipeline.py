import numpy
import math
from matplotlib import pyplot
from scipy import constants
import pyfits
from astropy.io import fits
import os

# FUNCTIONS
# --------------------------------------------------------------------------
def fitsconverter(loc,stem,freq):
    imhead = fits.open(loc+"/"+stem+'Q'+str(freq)+'.FITS')
    Freq = imhead[0].header['CRVAL3']
    Npix = imhead[0].header['NAXIS1']
    Q_im = fits.getdata(loc+"/"+stem+'Q'+str(freq)+'.FITS')[0,0,:,:]
    U_im = fits.getdata(loc+"/"+stem+'U'+str(freq)+'.FITS')[0,0,:,:]
    return Freq,Npix,Q_im,U_im

def transpose_multiplier(small_array,big_array):
    product = numpy.transpose(numpy.multiply(small_array,numpy.transpose(big_array)))
    return(product)

def double_transpose_multiplier(array1,array2):
    product = numpy.transpose(numpy.multiply(numpy.transpose(array1),numpy.transpose(array2)))
    return(product)

def display_array(array,outname):
    pyplot.imshow(array)
    pyplot.savefig(outname+'.png')

def Find_RMS(Map,Box):
    RMS=1.8*numpy.std(Map[(Box[0]):(Box[2]),(Box[1]):(Box[3])])
    return(RMS)

def propagate_PANG_error(Q,U,SigmaQ,SigmaU):
    # from Eoin/Seb's code, using standard error propagation technique
    # nb check if numpy has a function that can do this automatically!
    Part1=1/(4*numpy.add(numpy.multiply(Q,Q),numpy.multiply(U,U)))
    Part2=numpy.sqrt(numpy.add(numpy.multiply(numpy.multiply(Q,Q),SigmaU*SigmaU),numpy.multiply(numpy.multiply(U,U),SigmaQ*SigmaQ)))
    Error=numpy.multiply(Part1,Part2)*180/math.pi
    Weight=1/(numpy.multiply(Error,Error))
    return(Error,Weight)

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

    return(RM,eRM,Chi2,aW,aWxy,aWx,aWy,aWxx,D,Int)


# --------------------------------------------------------------------------


# INITIALIZATION
# --------------------------------------------------------------------------
stem = '0735+178_'
frequencies = 4
Shift = [1,1,1,1]
rms_box = [60,100,95,200]
subset = [400,400,600,600]
# [blcy,blcx,trcy,trcx]

# initialize empty arrays
Qs = []
Us = []
Qs_untrimmed = []
Us_untrimmed = []
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

    Freq,Npix,Q_array,U_array=fitsconverter(fits_loc,stem,i)
    Wavel = numpy.divide(constants.c,Freq)

    Q_rms = Find_RMS(Q_array,rms_box)
    U_rms = Find_RMS(U_array,rms_box)

    Q_array_trim = Q_array[subset[0]:subset[2],subset[1]:subset[3]]
    U_array_trim = U_array[subset[0]:subset[2],subset[1]:subset[3]]

    PANG_map = numpy.multiply(0.5*numpy.arctan2(U_array_trim,Q_array_trim),180/math.pi)
    PANG_Error, PANG_Sigma = propagate_PANG_error(Q_array_trim,U_array_trim,Q_rms,U_rms)

    wavelengths.append(Wavel*Wavel)
    Qs.append(Q_array_trim)
    Us.append(U_array_trim)
    Qs_untrimmed.append(Q_array)
    Us_untrimmed.append(U_array)
    Q_rmss.append(Q_rms)
    U_rmss.append(U_rms)
    PANGS.append(PANG_map)
    Errors.append(PANG_Error)
    Sigmas.append(PANG_Sigma)

# --------------------------------------------------------------

# convert lists of 2d arrays into 3d arrays
PANGS_arr = numpy.stack(PANGS)
Errors_arr = numpy.stack(Errors)
Sigma_arr = numpy.stack(Sigmas)
wavelengths_arr = numpy.array([wavelengths])

# --------------------------------------------------------------

def for_loop():

    # initialize RM arrays
    RM_map = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))
    eRM_map = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))

    Sigmas_summed = numpy.sum(Sigma_arr,axis=0)
    Wave_Pang_Sigma_summed = numpy.sum(numpy.multiply(Sigma_arr,transpose_multiplier(wavelengths_arr,PANGS_arr)),axis=0)
    Wave_Sigma_summed = numpy.sum(transpose_multiplier(wavelengths_arr,Sigma_arr),axis=0)
    Pang_Sigma_summed = numpy.sum(numpy.multiply(PANGS_arr,Sigma_arr),axis=0)
    Wave_Wave_sigma_summed = numpy.sum(transpose_multiplier(numpy.multiply(wavelengths_arr,wavelengths_arr),Sigma_arr),axis=0)

    Divider = numpy.multiply(Sigmas_summed,Wave_Wave_sigma_summed) - numpy.multiply(Wave_Sigma_summed,Wave_Sigma_summed)
    RM = (Sigmas_summed*Wave_Pang_Sigma_summed - Wave_Sigma_summed*Pang_Sigma_summed)/Divider
    Int_mat = (Pang_Sigma_summed*Wave_Wave_sigma_summed - Wave_Pang_Sigma_summed*Wave_Sigma_summed)/Divider

    Chi = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))
    eRM = numpy.zeros((subset[2]-subset[0],subset[3]-subset[1]))

    for i in range(0,frequencies):

        RM_term = numpy.add(Int_mat,(RM*wavelengths[i]))
        square_term_Chi = numpy.subtract(PANGS_arr[i],RM_term)
        Chi = numpy.add(Chi,numpy.square(square_term_Chi))

        sigma_term_1 = (numpy.multiply(Sigmas_summed,Sigma_arr[i]))*wavelengths[i]
        sigma_term_2 = numpy.multiply(Sigma_arr[i],Wave_Sigma_summed)
        square_term_eRM = numpy.divide((numpy.subtract(sigma_term_1,sigma_term_2)),Divider)

        eRM = eRM + numpy.sqrt(numpy.multiply(numpy.square(Errors_arr[i]),numpy.square(square_term_eRM)))

    RM_map = RM*(math.pi/180)
    eRM_map = eRM*(math.pi/180)

    # # --------------------------------------------------------------

    # fix!!!

    # populate RM arrays
    for i in range(0,subset[2]-subset[0]):
        for j in range(0,subset[3]-subset[1]):

            # get single pixel values
            PANG_pix = PANGS_arr[:,i,j]
            Errors_pix = Errors_arr[:,i,j]
            Sigma_pix = Sigma_arr[:,i,j]

            # initialize shifted values
            PANG_pix_shifted=numpy.zeros(frequencies)
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

                            RM_pix,eRM_pix,CHI_pix,aW,aWxy,aWx,aWy,aWxx,D,Int = RM_Calc(wavelengths,PANG_pix_shifted,Sigma_pix,Errors_pix,4)
                            print(CHI_pix)
                            if CHI_pix<CHI_pix_shifted:
                                RM_shifted=RM_pix
                                eRM_shifted=eRM_pix
                                CHI_pix_shifted=CHI_pix

            RM_shifted = float(RM_shifted)*math.pi/180
            eRM_shifted = float(eRM_shifted)*math.pi/180

            RM_map[i,j]=RM_shifted
            eRM_map[i,j]=eRM_shifted

    numpy.savetxt(txtfile_dest+'/eRMtest_loop_premask.txt',eRM_map)
    numpy.savetxt(txtfile_dest+'/RMtest_loop_premask.txt',RM_map)

    RM_map,eRM_map = Mask_Map(RM_map,eRM_map,10,subset[2]-subset[0],subset[3]-subset[1],50)

    numpy.savetxt(txtfile_dest+'/eRMtest_loop_postmask.txt',eRM_map)
    numpy.savetxt(txtfile_dest+'/RMtest_loop_postmask.txt',RM_map)

    return RM_map, eRM_map, eRM

# ----------------------------------------------------

def shift_vectors():

    Shift_Permutations = []

    for a in range(-Shift[0],Shift[0]+1):
        for b in range(-Shift[1],Shift[1]+1):
            for c in range(-Shift[2],Shift[2]+1):
                for d in range(-Shift[3],Shift[3]+1):
                    Shift_applied = [float(a)*180.0,float(b)*180.0,float(c)*180.0,float(d)*180.0]
                    Shift_Permutations.append(Shift_applied)

    Shift_Permutations_arr = numpy.stack(Shift_Permutations)
    Shifts = len(Shift_Permutations_arr)

    PANGS_stacked = numpy.broadcast_to(PANGS_arr,(Shifts,frequencies,subset[2]-subset[0],subset[3]-subset[1]))
    PANGS_shifted = numpy.transpose(numpy.add(numpy.transpose(numpy.float32(Shift_Permutations_arr)),numpy.transpose(PANGS_stacked)))

    wavelengths_arr_stacked = numpy.broadcast_to(wavelengths_arr,(Shifts,frequencies))
    Sigma_arr_stacked = numpy.broadcast_to(Sigma_arr,(Shifts,frequencies,subset[2]-subset[0],subset[3]-subset[1]))

    Sigmas_summed_stacked = numpy.sum(Sigma_arr_stacked,axis=1)
    Wave_Pang_Sigma_summed_stacked = numpy.sum(numpy.multiply(Sigma_arr_stacked,double_transpose_multiplier(wavelengths_arr_stacked,PANGS_shifted)),axis=1)
    Wave_Sigma_summed_stacked = numpy.sum(double_transpose_multiplier(wavelengths_arr_stacked,Sigma_arr_stacked),axis=1)
    Pang_Sigma_summed_stacked = numpy.sum(numpy.multiply(PANGS_shifted,Sigma_arr_stacked),axis=1)
    Wave_Wave_sigma_summed_stacked = numpy.sum(double_transpose_multiplier(numpy.multiply(wavelengths_arr_stacked,wavelengths_arr_stacked),Sigma_arr_stacked),axis=1)

    Divider_stacked = numpy.multiply(Sigmas_summed_stacked,Wave_Wave_sigma_summed_stacked) - numpy.multiply(Wave_Sigma_summed_stacked,Wave_Sigma_summed_stacked)
    RM_stacked = numpy.divide((numpy.multiply(Sigmas_summed_stacked,Wave_Pang_Sigma_summed_stacked) - numpy.multiply(Wave_Sigma_summed_stacked,Pang_Sigma_summed_stacked)),Divider_stacked)
    Int_stacked = numpy.divide(numpy.multiply(Pang_Sigma_summed_stacked,Wave_Wave_sigma_summed_stacked) - numpy.multiply(Wave_Pang_Sigma_summed_stacked,Wave_Sigma_summed_stacked),Divider_stacked)

    RM_stacked_four = numpy.stack([RM_stacked,RM_stacked,RM_stacked,RM_stacked],axis=1)
    Int_stacked_four = numpy.stack([Int_stacked,Int_stacked,Int_stacked,Int_stacked],axis=1)

    RM_wav = numpy.transpose(numpy.multiply(numpy.transpose(RM_stacked_four),numpy.transpose(wavelengths_arr)))
    RM_term_stacked = numpy.add(Int_stacked_four,RM_wav)
    square_term_Chi_stacked = numpy.subtract(PANGS_shifted,RM_term_stacked)

    Chi_stacked = numpy.sum(numpy.square(square_term_Chi_stacked),axis=1)

    print(Chi_stacked[56,0,0])

    sigma_summed_stacked_four = numpy.stack([Sigmas_summed_stacked,Sigmas_summed_stacked,Sigmas_summed_stacked,Sigmas_summed_stacked],axis=1)
    sigma_term_1_stacked = double_transpose_multiplier(wavelengths_arr_stacked,numpy.multiply(sigma_summed_stacked_four,Sigma_arr_stacked))
    Wave_Sigma_summed_stacked_four = numpy.stack([Wave_Sigma_summed_stacked,Wave_Sigma_summed_stacked,Wave_Sigma_summed_stacked,Wave_Sigma_summed_stacked],axis=1)
    sigma_term_2_stacked = numpy.multiply(Sigma_arr_stacked,Wave_Sigma_summed_stacked_four)
    subtract_term = numpy.subtract(sigma_term_1_stacked,sigma_term_2_stacked)
    Divider_stacked_four = numpy.stack([Divider_stacked,Divider_stacked,Divider_stacked,Divider_stacked],axis=1)
    square_term_eRM_stacked = numpy.divide(subtract_term,Divider_stacked_four)


    Errors_stacked = numpy.broadcast_to(Errors_arr,(Shifts,frequencies,subset[2]-subset[0],subset[3]-subset[1]))
    eRM_stacked = numpy.sum(numpy.sqrt(numpy.multiply(numpy.square(Errors_stacked),numpy.square(square_term_eRM_stacked))),axis=1)


    index_array = numpy.argmin(Chi_stacked,axis=0)

    RM = numpy.take_along_axis(RM_stacked, numpy.expand_dims(index_array, axis=0), axis=0)
    eRM = numpy.take_along_axis(eRM_stacked, numpy.expand_dims(index_array, axis=0), axis=0)

    RM = RM[0,:,:]
    eRM = eRM[0,:,:]

    RM_map = RM*(math.pi/180)
    eRM_map = eRM*(math.pi/180)

    numpy.savetxt(txtfile_dest+'/eRMtest_stack_premask.txt',eRM_map)
    numpy.savetxt(txtfile_dest+'/RMtest_stack_premask.txt',RM_map)

    RM_map,eRM_map = Mask_Map(RM_map,eRM_map,10,subset[2]-subset[0],subset[3]-subset[1],50)

    numpy.savetxt(txtfile_dest+'/eRMtest_stack_postmask.txt',eRM_map)
    numpy.savetxt(txtfile_dest+'/RMtest_stack_postmask.txt',RM_map)

    return RM_map, eRM_map,eRM_stacked

# ----------------------------------------------------

RM_map_stack,eRM_map_stack,test_stack = shift_vectors()

plot_map(RM_map_stack,'RMmapstack',png_dest)
plot_map(eRM_map_stack,'eRMmapstack',png_dest)
plot_map(test_stack[0,:,:],'teststack',png_dest)

#
# fig = pyplot.figure()
#
# for i in range(0,len(wavelengths)):
#
#     # print(Freq,Npix)
#     # print(Q_rms,U_rms)
#
#     ax_q = fig.add_subplot(3,4,i+1)
#     Q_arr = numpy.flipud(Qs_untrimmed[i])
#     ax_q.imshow(Q_arr)
#     ax_q.plot([rms_box[0],rms_box[2],rms_box[2],rms_box[0],rms_box[0]],\
#             [rms_box[1],rms_box[1],rms_box[3],rms_box[3],rms_box[1]])
#
#     ax_u = fig.add_subplot(3,4,i+5)
#     U_arr = numpy.flipud(Us_untrimmed[i])
#     ax_u.imshow(U_arr)
#     ax_u.plot([rms_box[0],rms_box[2],rms_box[2],rms_box[0],rms_box[0]],\
#             [rms_box[1],rms_box[1],rms_box[3],rms_box[3],rms_box[1]])
#
#     ax_pang = fig.add_subplot(3,4,i+9)
#     pang_arr = numpy.flipud(PANGS_arr[i])
#     ax_pang.imshow(pang_arr)
#
#
# fig.savefig(png_dest+"/"+'test.png')
