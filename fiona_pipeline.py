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
    # get all required data and info from fitsfiles
    imhead = fits.open(loc+"/"+stem+'Q'+str(freq)+'.FITS')
    Freq = imhead[0].header['CRVAL3']
    Wavel = numpy.divide(constants.c,Freq)
    Npix = imhead[0].header['NAXIS1']
    Q_im = fits.getdata(loc+"/"+stem+'Q'+str(freq)+'.FITS')[0,0,:,:]
    U_im = fits.getdata(loc+"/"+stem+'U'+str(freq)+'.FITS')[0,0,:,:]
    return Freq,Wavel,Npix,Q_im,U_im

def double_transpose_multiplier(array1,array2):
    product = numpy.transpose(\
                numpy.multiply(\
                    numpy.transpose(array1),\
                    numpy.transpose(array2)))
    return(product)

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

def permute(list,mod=1):

    permutations = []

    for a in range(-list[0],list[0]+1):
        for b in range(-list[1],list[1]+1):
            for c in range(-list[2],list[2]+1):
                for d in range(-list[3],list[3]+1):
                    permutation = [float(a)*mod,float(b)*mod,float(c)*mod,float(d)*mod]
                    permutations.append(permutation)

    permutations_arr = numpy.stack(permutations)
    return permutations_arr

def gen_sums(S,Wav,P,Er):

    # add sigmas together at each frequency
    SSS = numpy.sum(S,axis=1)

    # multiply wav^2 by PANG, then multiply by sigma, then sum over freqs
    WPSSS = numpy.sum(\
                numpy.multiply(\
                    S,double_transpose_multiplier(Wav,P)),\
            axis=1)

    # multiply wav^2 by sigma, sum over frequencies
    WSSS = numpy.sum(\
            double_transpose_multiplier(Wav,S),\
           axis=1)

    # multiply pang by sigma, sum over frequencies
    PSSS = numpy.sum(numpy.multiply(P,S),axis=1)

    # multiply wav^2 by wav^2, then multiply by sigma
    WWSSS = numpy.sum(\
                double_transpose_multiplier(\
                    numpy.multiply(Wav,Wav),\
                    S),\
            axis=1)

    ESS = numpy.sum(Er,axis=1)
    PSS = numpy.sum(P,axis=1)

    sum_dict = {"sigma_summed":SSS,\
                "wave_pang_sigma_summed":WPSSS,\
                "wave_sigma_summed":WSSS,\
                "pang_sigma_summed":PSSS,\
                "wave_wave_sigma_summed":WWSSS,\
                "Error_summed":ESS,\
                "Pang_summed":PSS}

    return sum_dict

def gen_RM(sum_dict,S,Wav,P,Er):

    SSS = sum_dict["sigma_summed"]
    WPSSS = sum_dict["wave_pang_sigma_summed"]
    WSSS = sum_dict["wave_sigma_summed"]
    PSSS = sum_dict["pang_sigma_summed"]
    WWSSS = sum_dict["wave_wave_sigma_summed"]
    ESS = sum_dict["Error_summed"]
    PSS = sum_dict["Pang_summed"]

    Divider_stacked = numpy.multiply(SSS,WWSSS) - numpy.multiply(WSSS,WSSS)

    RM_stacked = numpy.divide((numpy.multiply(SSS,WPSSS) - numpy.multiply(WSSS,PSSS)),Divider_stacked)
    Int_stacked = numpy.divide(numpy.multiply(PSSS,WWSSS) - numpy.multiply(WPSSS,WSSS),Divider_stacked)

    # calculate this NOT summed over freq and then sum after
    Err_prod = numpy.multiply(Er,Er)
    Sigma_Wav_prod = double_transpose_multiplier(Wav,S)
    term_a = numpy.transpose(numpy.multiply(numpy.transpose(Sigma_Wav_prod,(1,0,2,3)),SSS),(1,0,2,3))
    term_b = numpy.transpose(numpy.multiply(numpy.transpose(S,(1,0,2,3)),WSSS),(1,0,2,3))

    # eRM+numpy.sqrt(S[i]*S[i]*((aW*W[i]*X[i]-W[i]*aWx)/D)*((aW*W[i]*X[i]-W[i]*aWx)/D))
    topline = numpy.subtract(term_a,term_b)
    divided_square = numpy.square(numpy.transpose(numpy.divide(numpy.transpose(topline,(1,0,2,3)),Divider_stacked),(1,0,2,3)))
    eRM = numpy.sqrt(numpy.multiply(Err_prod,divided_square))
    eRM_summed = numpy.sum(eRM,axis=1)
    # Chi2=Chi2+(Y[i]-(RM*X[i]+Int))*(Y[i]-(RM*X[i]+Int))

    RM_stacked_four = stack_the_same(RM_stacked,1)
    Int_stacked_four = stack_the_same(Int_stacked,1)
    Divider_stacked_four = stack_the_same(Divider_stacked,1)



    square_term = numpy.subtract(P,numpy.add(double_transpose_multiplier(Wav,RM_stacked_four),Int_stacked_four))
    Chi = numpy.square(square_term)
    Chi_summed = numpy.sum(Chi,axis=1)

    return RM_stacked,Int_stacked,Divider_stacked,eRM_summed,Chi_summed

def stack_the_same(array,ax):
    stacked = numpy.stack([array,array,array,array],axis=ax)
    return stacked

# --------------------------------------------------------------------------
# INITIALIZATION
# --------------------------------------------------------------------------
stem = '0735+178_'
frequencies = 4
Shift = [1,1,1,1]
rms_box = [60,100,95,200]
subset = [400,400,600,600]
range_x = subset[2]-subset[0]
range_y = subset[3]-subset[1]
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

    Freq,Wavel,Npix,Q_array,U_array=fitsconverter(fits_loc,stem,i)

    Q_rms = Find_RMS(Q_array,rms_box)
    U_rms = Find_RMS(U_array,rms_box)

    Q_array_trim = Q_array[subset[0]:subset[2],subset[1]:subset[3]]
    U_array_trim = U_array[subset[0]:subset[2],subset[1]:subset[3]]

    PANG_map = numpy.multiply(0.5*numpy.arctan2(U_array_trim,Q_array_trim),180/math.pi)
    PANG_Error, PANG_Sigma = propagate_PANG_error(Q_array_trim,U_array_trim,Q_rms,U_rms)
    # sigma, weight
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

def shift_vectors():

    # -----------------------------------------------

    Shift_Permutations_arr = permute(Shift,180)
    Shifts = len(Shift_Permutations_arr)

    PANGS_stacked = numpy.broadcast_to(PANGS_arr,(Shifts,frequencies,range_x,range_y))
    wavelengths_arr_stacked = numpy.broadcast_to(wavelengths_arr,(Shifts,frequencies))
    Sigma_arr_stacked = numpy.broadcast_to(Sigma_arr,(Shifts,frequencies,range_x,range_y))
    Errors_arr_stacked = numpy.broadcast_to(Errors_arr,(Shifts,frequencies,range_x,range_y))

    PANGS_shifted = numpy.transpose(\
                        numpy.add(\
                            numpy.transpose(numpy.float32(Shift_Permutations_arr)),\
                            numpy.transpose(PANGS_stacked)))

    # -----------------------------------------------

    sum_dict = gen_sums(Sigma_arr_stacked,wavelengths_arr_stacked,PANGS_shifted,Errors_arr_stacked)

    RM_stacked,\
    Int_stacked,\
    Divider_stacked,\
    eRM_stacked,\
    Chi_stacked = gen_RM(sum_dict,Sigma_arr_stacked,wavelengths_arr_stacked,PANGS_shifted,Errors_arr_stacked)

    # -----------------------------------------------

    # sigma_summed_stacked_four = stack_the_same(sum_dict["sigma_summed"],1)
    # Wave_Sigma_summed_stacked_four = stack_the_same(sum_dict["wave_sigma_summed"],1)
    #
    # RM_stacked_four = stack_the_same(RM_stacked,1)
    # Int_stacked_four = stack_the_same(Int_stacked,1)
    # Divider_stacked_four = stack_the_same(Divider_stacked,1)
    #
    # RM_wav = numpy.transpose(\
    #             numpy.multiply(\
    #                 numpy.transpose(RM_stacked_four),
    #                 numpy.transpose(wavelengths_arr)))
    #
    # RM_term_stacked = numpy.add(\
    #                     Int_stacked_four,\
    #                     RM_wav)
    #
    # square_term_Chi_stacked = numpy.subtract(\
    #                             PANGS_shifted,\
    #                             RM_term_stacked)
    #
    # Chi_stacked = numpy.sum(\
    #                 numpy.square(square_term_Chi_stacked),axis=1)
    #
    #
    # sigma_term_1_stacked = double_transpose_multiplier(\
    #                         wavelengths_arr_stacked,\
    #                         numpy.multiply(\
    #                             sigma_summed_stacked_four,
    #                             sigma_summed_stacked_four))
    #
    #
    # sigma_term_2_stacked = numpy.multiply(\
    #                         Sigma_arr_stacked,
    #                         Wave_Sigma_summed_stacked_four)
    #
    # subtract_term = numpy.subtract(\
    #                     sigma_term_1_stacked,
    #                     sigma_term_2_stacked)
    #
    #
    # square_term_eRM_stacked = numpy.divide(\
    #                             subtract_term,
    #                             Divider_stacked_four)
    #
    # eRM_stacked = numpy.sum(\
    #                 numpy.sqrt(\
    #                     numpy.multiply(\
    #                         numpy.square(Errors_arr_stacked),\
    #                         numpy.square(square_term_eRM_stacked))),\
    #                 axis=1)
    index_array = numpy.argmin(Chi_stacked,axis=0)

    RM = numpy.take_along_axis(\
            RM_stacked,\
            numpy.expand_dims(\
                index_array,\
            axis=0),\
         axis=0)

    eRM = numpy.take_along_axis(\
            eRM_stacked,\
            numpy.expand_dims(\
                index_array,\
            axis=0),\
          axis=0)

    RM = RM[0,:,:]
    eRM = eRM[0,:,:]
#
    RM_map = RM*(math.pi/180)
    eRM_map = eRM*(math.pi/180)
#
    RM_map,eRM_map = Mask_Map(RM_map,eRM_map,10,range_x,range_y,50)
#
    return RM_map, eRM_map,eRM_stacked
#
# # ----------------------------------------------------
#
RM_map_stack,eRM_map_stack,test_stack = shift_vectors()
#
plot_map(RM_map_stack,'RMmapstack',png_dest)
plot_map(eRM_map_stack,'eRMmapstack',png_dest)
plot_map(test_stack[0,:,:],'teststack',png_dest)

#
fig = pyplot.figure()
#
# for i in range(0,len(wavelengths)):
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
