from __future__ import print_function

import cv2
import os
import numpy as np
import cv2 as cv
import argparse
import sys




def video_size(filename):
    # Path to the video file
    video_file = filename

    # Get the size of the video file in bytes
    video_size = os.path.getsize(video_file)

    # Print the size of the video file in bytes
    # print(f"Video size: {video_size} bytes")
    return video_size

def bitrate(filename):
    # Use a breakpoint in the code line below to debug your script.
    cap = cv2.VideoCapture(filename)
    br = int(cap.get(cv2.CAP_PROP_BITRATE))
    print("Video "+filename+ f"bitrate: {br} bps")
    cap.release()
    cv2.destroyAllWindows()

def frames(filename):
    # Use a breakpoint in the code line below to debug your script.
    cap = cv2.VideoCapture(filename)
    br = int(cap.get(cv2.CAP_PROP_FRAME_COUNT ))
    print(f"Video frames: {br} frames")
    cap.release()
    cv2.destroyAllWindows()


def compression_ratio(filename):
    avi='videos/sample_1280x720.avi'
    # Use a breakpoint in the code line below to debug your script.
    return "avi to " + filename+ " compression ratio: "+str(round(video_size(filename)/video_size(avi)*100,2))+"%"


# [get-psnr]
def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr
# [get-psnr]
# [get-mssim]
def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    return mssim
# [get-mssim]
def image_quality_comparisson(filename):

    sourceReference = "videos/sample_1280x720.avi"  #Path to reference video
    sourceCompareWith = filename  #Path to the video to be tested
    delay = 3 #Time delay
    psnrTriggerValue = 30 #PSNR Trigger Value
    framenum = -1 # Frame counter
    captRefrnc = cv.VideoCapture(cv.samples.findFileOrKeep(sourceReference))
    captUndTst = cv.VideoCapture(cv.samples.findFileOrKeep(sourceCompareWith))
    if not captRefrnc.isOpened():
        print("Could not open the reference " + sourceReference)
        sys.exit(-1)
    if not captUndTst.isOpened():
        print("Could not open case test " + sourceCompareWith)
        sys.exit(-1)
    refS = (int(captRefrnc.get(cv.CAP_PROP_FRAME_WIDTH)), int(captRefrnc.get(cv.CAP_PROP_FRAME_HEIGHT)))
    uTSi = (int(captUndTst.get(cv.CAP_PROP_FRAME_WIDTH)), int(captUndTst.get(cv.CAP_PROP_FRAME_HEIGHT)))
    if refS != uTSi:
        print("Inputs have different size!!! Closing.")
        sys.exit(-1)
    WIN_UT = "Under Test"
    WIN_RF = "Reference"
    cv.namedWindow(WIN_RF, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(WIN_UT, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(WIN_RF, 400, 0) #750,  2 (bernat =0)
    cv.moveWindow(WIN_UT, refS[0], 0) #1500, 2
    print("Reference frame resolution: Width={} Height={} of nr#: {}".format(refS[0], refS[1],
                                                                             captRefrnc.get(cv.CAP_PROP_FRAME_COUNT)))
    print("PSNR trigger value {}".format(psnrTriggerValue))
    while True: # Show the image captured in the window and repeat
        _, frameReference = captRefrnc.read()
        _, frameUnderTest = captUndTst.read()
        if frameReference is None or frameUnderTest is None:
            print(" < < <  Video over!  > > > ")
            break
        framenum += 1
        psnrv = getPSNR(frameReference, frameUnderTest)
        print("Frame: {}# : PSNR: {}dB,".format(framenum, round(psnrv, 3)), end=" ")
        if (psnrv < psnrTriggerValue and psnrv):
            mssimv = getMSSISM(frameReference, frameUnderTest)
            print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[2] * 100, 2), round(mssimv[1] * 100, 2),
                                                     round(mssimv[0] * 100, 2)), end=" ")
        print()
        cv.imshow(WIN_RF, frameReference)
        cv.imshow(WIN_UT, frameUnderTest)
        k = cv.waitKey(delay)
        if k == 27:
            break


if __name__ == '__main__':
    path = "C:\\Users\Sergiu\\PycharmProjects\\exerciu_1_cav\\exerciu_1_cav\\videos"
    vids = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    with open('rezultate_exercitiu_1.txt', 'w') as file:
        sys.stdout = file
        # Bitrate

        print("\n")
        print("Bitrate\n")
        for v in vids:
            bitrate('videos/' + v)

        vids.remove("sample_1280x720.avi")
        print("\n")
        print(200 * "*" + "\n")
        print("Compression ratio\n")

        # Compression ratio

        for v in vids:
            print(compression_ratio('videos/' + v))

        print("\n")
        print(200 * "*" + "\n")
        print("Quality comparisson\n")

        # Quality comparisson - aici am pornit de la un template gasit in documentatia celor de opencv de comparat doua
        # video-uri si daca se va rula codul vom putea observa de ca diferenta nr de frame-uri intre videoclipuri
        # influenteaza valoarea PSNR  intrucat nu comparam exact acelasi lucru
        # totodata avand in vedere ca PNSR e predominat sub 30 dB putem afirma ca procesarea nu e ridicata( high)

        for v in vids:
            print("\n")
            print(200 * "*" + "\n")
            print("Quality comparisson for " + v + "\n")
            image_quality_comparisson('videos/' + v)

        sys.stdout = sys.__stdout__










