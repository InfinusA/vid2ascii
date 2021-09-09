#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import sys
import time
from pprint import pprint
from functools import reduce
from math import gcd
from multiprocessing import Queue

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cf(num1,num2):
    n=[]
    g=gcd(num1, num2)
    for i in range(1, g+1): 
        if g%i==0: 
            n.append(i)
    return n

aparse = argparse.ArgumentParser("Vid2Ascii2")
aparse.add_argument('-o', '--output', default=None) #input file but edited name
aparse.add_argument('-s', '--scale', default=16, type=int)
aparse.add_argument('-n', '--font',
    default='/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf')
aparse.add_argument('--fontsize', default=10, type=int)
aparse.add_argument('-t', '-j', '--threads', default=None, type=int) #all available threads
aparse.add_argument('-k', '--frameskip', default=0, type=int)
aparse.add_argument('-r', '--resolution', default=None,
    type=lambda x: [int(e) for e in x.split('x')])
    #default resolution in 1x1 style of writing resolutionz
aparse.add_argument('-b', '--blocksize', default=8, type=int) #block size
aparse.add_argument('-c', '--charset', default=' .:-+=*#%@')
aparse.add_argument('-e', '--encoding', default="DIVX")
aparse.add_argument("inputfile")
#['-s8','/home/infinusa/Videos/Sword Art Online II - Opening 1 HD (Creditless)-8nF8PBl1KsM.mkv']
progargs = aparse.parse_args()
if progargs.output is None:
    if progargs.inputfile.rfind('.') != -1:
        strippedname = progargs.inputfile[:progargs.inputfile.rfind('.')]
    else:
        strippedname = progargs
    progargs.output = strippedname + "_ascii.avi"
    print(strippedname)

if progargs.threads is None:
    progargs.threads = mp.cpu_count()

#init all variables
video_font = ImageFont.truetype(progargs.font, progargs.fontsize)
video_fontscale = [6, 7]
# video_fontscale = list(list(video_font.getsize(" ")))
# video_fontscale[0] = video_fontscale[0]*2
# video_fontscale = (5, 5)
print(f"Font size: {video_fontscale}")
video_source = cv2.VideoCapture(progargs.inputfile)
video_fps = video_source.get(cv2.CAP_PROP_FPS)
video_framecount = video_source.get(cv2.CAP_PROP_FRAME_COUNT)
video_size = (int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_sizescaled = (int(round(video_size[0]/progargs.scale)), int(round(video_size[1]/progargs.scale)))
output_fps = video_fps / (progargs.frameskip+1)
output_size = (int(round(video_sizescaled[0]*video_fontscale[0])), int(round(video_sizescaled[1]*video_fontscale[1]))+1)
print(output_size)
output_fourcc = cv2.VideoWriter_fourcc(*progargs.encoding)
output_source = cv2.VideoWriter(progargs.output, output_fourcc, output_fps, output_size)


def frame_iter2(frame):
    newx = frame.shape[0]*video_fontscale[0]
    newy = frame.shape[1]*video_fontscale[1]
    nframe = Image.new("RGB", (newx, newy), color=(0, 0, 0))
    draw = ImageDraw.Draw(nframe)
    frame = np.vectorize(lambda x: progargs.charset[x])(np.rint((frame/255)*(len(progargs.charset)-1)).astype(int))
    # frame = np.repeat(frame, 2, axis=1)
    frame = np.insert(frame, frame.shape[1],'\n',axis=1)
    outstr = "".join(frame.flatten())
    draw.text((0, 0),outstr,(255,255,255),font=video_font)
    return cv2.cvtColor(np.array(nframe), cv2.COLOR_RGB2BGR)

def frame_itest(frame):
    newx = frame.shape[0]*video_fontscale[0]
    newy = frame.shape[1]*video_fontscale[1]
    rs = cv2.resize(frame, (newx, newy))
    # print(rs.shape)
    # print(progargs.blocksize * video_fontscale[0], progargs.blocksize * video_fontscale[1])
    return cv2.cvtColor(rs, cv2.COLOR_GRAY2BGR)

def queue_processor(data):
    #output_queue.put(data)
    framepos = data[0]
    framedata = frame_iter2(data[1])#cv2.cvtColor(data[1], cv2.COLOR_GRAY2BGR)#frame_iter2(data[1])
    return framepos, framedata

with mp.Pool(processes=progargs.threads) as threadpool:
    print('Preparing threads')
    framenumber = -1
    completed = False
    # threadpool.starmap_async(queue_processor, [(input_queue, output_queue) for _ in range(progargs.threads)])

    # print('Making sure threads are working')
    # for _ in range(progargs.threads):
    #     output_queue.get()

    while True:
        framenumber += 1
        print(f"Current frame: {framenumber}")
        success, baseframe = video_source.read()
        if not success:
            break
        if framenumber % (progargs.frameskip + 1) != 0:
            print(f'skipping frame no: {framenumber}')
            continue
        scaledframe = cv2.resize(baseframe, video_sizescaled)
        grayframe = cv2.cvtColor(scaledframe, cv2.COLOR_BGR2GRAY)
        
        #testframe = np.zeros(grayframe.shape, np.uint8)
        testframe = np.zeros((*reversed(output_size), 3), np.uint8)

        #NOTE: The first index of an image recurses vertically, and the second index recurses horizontally
        chunkdict = {}
        #xblocks = (len(grayframe)//progargs.blocksize) + len(grayframe)%progargs.blocksize
        xsplit = []# = np.array_split(grayframe, xblocks, axis=1)
        xiter = 0
        while True:
            dat = grayframe[xiter*progargs.blocksize:(xiter+1)*progargs.blocksize]
            
            if all(e != 0 for e in dat.shape):
                xsplit.append(dat)
            elif len(dat) != progargs.blocksize:
                break
            else:
                dat = grayframe[xiter*progargs.blocksize:]
                if all(e != 0 for e in dat.shape):
                    xsplit.append(dat)
                break
            xiter += 1
    
        #assert len(xsplit[0]) == progargs.blocksize
        #check to make sure the xsplit joined together is the same as the original
        assert (grayframe == np.vstack(xsplit)).all()
        #split the dict vertically. They are now very tall, and in order, go from left to right
        #yblocks = (len(xsplit[0][0])//progargs.blocksize)# + len(xsplit[0][0])%progargs.blocksize
        tys = []
        for xindex, xitem in enumerate(xsplit):
            #vertically split each tall xitem
            yiter = 0
            ysplit = []
            while True:
                #[
                # :, - take all items on x-axis
                # yiter*progargs.blocksize: (yiter+1)*progargs.blocksize - from start of block to end of block
                # ]
                dat = xitem[:, yiter*progargs.blocksize: (yiter+1)*progargs.blocksize]
                if 0 in dat.shape:
                    #if the shape ends up being invalid
                    dat = xitem[:, yiter*progargs.blocksize:]
                    #dat = whatever's left in that vertical column
                
                if 0 not in dat.shape:
                    ysplit.append(dat)
                    yiter += 1
                else:
                    break
            #add the yitems to the dict of chunks
            for yindex, ychunk in enumerate(ysplit):
                if np.all(ychunk == 0):
                    continue
                chunkdict[(xindex, yindex)] = ychunk
            tys.append(ysplit)

        results = threadpool.map_async(queue_processor, chunkdict.items())

        for coords, chunk in results.get():#chunkdict.items():
            a=(progargs.blocksize * video_fontscale[1], progargs.blocksize * video_fontscale[0])
            if tuple(chunk.shape[0:2]) == (progargs.blocksize * video_fontscale[1], progargs.blocksize * video_fontscale[0]):
                xstart = coords[0] * progargs.blocksize * video_fontscale[1]
                xend = xstart + chunk.shape[0]
                ystart = coords[1] * progargs.blocksize * video_fontscale[0]
                yend = ystart + chunk.shape[1]
                assert np.all(testframe[xstart:xend,ystart:yend] == 0)
                # assert testframe[xstart:xend,ystart:yend].size != 0
                testframe[xstart:xend,ystart:yend] = chunk
                cv2.imshow("title", testframe)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    completed = True
                    break
            # else:
                # print('\n', coords, '\n')
                # print(chunk.shape)
                # print(progargs.blocksize * video_fontscale[0], progargs.blocksize * video_fontscale[1])
            #     cv2.imwrite(f"chunks/{framenumber}-{coords[1]}-{coords[0]}.png", chunk)
        output_source.write(testframe)
        if completed:
            break
    
video_source.release()
# output_source.release()
cv2.destroyAllWindows()