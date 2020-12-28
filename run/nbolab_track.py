"""
    Used for testing in non-pi, or with regular usb-cam
    nopi must point to the camera number to use
"""
import argparse

"""
    Read the arguments passed for program execution
"""
parserArg = argparse.ArgumentParser(description='Main software parameters')
parserArg.add_argument('--file_name', type=str, help='name of the file')
parserArg.add_argument('--background', type=str, help='specify the path of back')
parserArg.add_argument('--capture', type=int, help='set the camera stream vide0 as default')
parserArg.add_argument('--fps', type=int, help='set frames per second for processing')
parserArg.add_argument('--mode', type=str, help='set the execution mode')
parserArg.add_argument('--width', type=int, help='resize for faster processing')
parserArg.add_argument('--height', type=int, help='resize for faster processing')
parserArg.add_argument('-np', '--nopi', type=str, help='activate normal usb-cam mode')
parserArg.add_argument('-f', '--folder', type=str, help='folder with png, if not only png use regex')
args = parserArg.parse_args()

if args.nopi is not None:
    from tracking_functions import *
    import os
    import glob
    import csv
    import socket
    from configparser import ConfigParser
    from time import time as timer
    from pathlib import Path
    import datetime
    import time
else:
    from tracking_functions import *
    import os
    import glob
    import csv
    import socket
    import argparse
    from configparser import ConfigParser
    from time import time as timer
    from pathlib import Path
    import picamera
    import datetime
    import time
    from vidgear.gears import PiGear
"""
    Paths definition

    Home
    Repository
    Config
    Background
    Background file: bg_HOSTNAME.png
    csv files
    stream folder for preview
"""
home = str(Path.home())
repo = home + '/nbolab_track'
config = repo + '/config/config.conf'
backgrounds = repo + '/background'
hostname = os.popen('hostname').read().rstrip('\n')
csv_files = repo + '/csv_bak'
stream_folder = home + '/nfs' + '/' + str(os.uname()[1])

"""
    Read the parameters in the configuration file
    NOTE: file must by built with python
"""
parser = ConfigParser()
parser.read(config)

"""
    Set the arguments passed or set to defaults
"""
# defines the csv filename, uses hostname + date as default
if args.file_name is not None:
    label = '/' + str(args.file_name)
else:
    label = '/' + str(socket.gethostname()) + "_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# set the background, uses the background folder as default
if args.background is not None:
    bg = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
    print("User specified background")
    print("Background size:" + str(bg.shape))
else:
    bg = cv2.imread(backgrounds + '/bg_' + str(hostname) + '.png', cv2.IMREAD_GRAYSCALE)
    print("Background size:" + str(bg.shape) + " Loaded from: " + backgrounds + '/bg_' + str(hostname) + '.png')

# set fps control
"""
    not operational
"""
if args.fps is not None:
    fps = args.fps
    fps /= 1000
    print("user defined FPS: " + str(args.fps))
else:
    fps = 20
    fps /= 1000

# set the execution mode
if args.mode is not None:
    mode = args.mode
    print("Selected mode is: " + str(mode))
else:
    mode = 'experiment'
    print("Selected mode is: " + str(mode))

# set resize
if args.width and args.height is not None:
    resize = True
    width = args.width
    height = args.height
else:
    width = 640
    height = 480
    resize = False

# set the capture device
if args.nopi is not None:
    stream = cv2.VideoCapture(int(args.nopi))
    ret, test_frame = stream.read()
else:
    stream = PiGear(resolution=(640, 480), framerate=60, colorspace='COLOR_BGR2GRAY').start()
    test_frame = stream.read()

# set the folder for offline mode
img_files = glob.glob(args.folder)
if args.mode != 'offline':
    print("Foreground size: " + str(test_frame.shape))
else:
    test_frame = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
    print("Image size: " + str(test_frame.shape))
    width = test_frame.shape[0]
    height = test_frame.shape[1]
    print("width: " + str(width))
    print("height: " + str(height))

# start the empty canvas
if args.mode != 'offline':
    canvas = np.zeros((test_frame.shape[0], test_frame.shape[1]))
else:
    canvas = np.zeros((test_frame.shape[0], test_frame.shape[1]))

# set background
if resize is True:
    bg = cv2.resize(bg, (width, height), interpolation = cv2.INTER_AREA)
else:
    bg = bg
    

"""
    Program main loop

    Writes a csv with the points extracted from the body tracking functions
    
    All parameters are defined in the config file, and initialized here

"""

### PARAMETERS ###
d=parser.getint('preprocess', 'filter_size')
sigma1=parser.getint('preprocess', 'sigma_color')
sigma2=parser.getint('preprocess', 'sigma_space')
kx=parser.getint('postprocess', 'kernelx')
ky=parser.getint('postprocess', 'kernely')
print("Filter size: " + str(d))
print("Sigma1 size: " + str(sigma1))
print("Sigma2 size: " + str(sigma2))
print("Kernel size: " + str((kx, ky)))
### PARAMETERS END ###

if mode == 'experiment':

    with open(csv_files + label + '.csv', 'w') as f:

        ### CSV HEADERS ###
        writer = csv.writer(f)
        writer.writerow(["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "MICROSECOND", "centroid_x", "centroid_y", "tail_x", "tail_y", "head_x", "head_y"])
        ### CSV HEADERS END ###

        ### EXEC TIME CALC ### 
        start_time = time.time()
        ###                ###

        while True:

            # read a single frame
            if resize is True:
                if args.nopi is not None:
                    ret, frame = cv2.resize(stream.read(), (width, height), interpolation = cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.resize(stream.read(), (width, height), interpolation = cv2.INTER_LINEAR)
            else:
                if args.nopi is not None:
                    ret, frame = stream.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = stream.read()

            ### IMAGE PROCESSING ###
            frame_filter = preprocess_image(frame, d, sigma1, sigma2)
            frame_diff = bgfg_diff(bg, frame_filter)
            contours, area, n_contours = contour_extraction(frame_diff)
            frame_post = postprocess_image(contours, kx, ky)
            time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
            ### IMAGE PROCESSING END ###

            ### POINTS EXTRACTION ###
            if resize is True:
                area_ratio = area / (width * height)
            else:
                area_ratio = area / (frame.shape[0] * frame.shape[1])
            """
                Will only calculate points if:
                The number of contours detected is not 0 and
                The ratio of the largest contour is not greater than
                10% of whole image area
                Otherwise will write 'NA' and drop the frame from further processing
            """
            if n_contours != 0 and area_ratio < 0.1:
                M = cv2.moments(frame_post)
                centroidX = int(M['m10'] / M['m00'])
                centroidY = int(M['m01'] / M['m00'])
                tailX = 0
                tailY = 0
                headX = 0
                headY = 0
            else:
                centroidX = 'NA'
                centroidY = 'NA'
                tailX = 'NA'
                tailY = 'NA'
                headX = 'NA'
                headY = 'NA'
            ### POINTS EXTRACTION END ###


            ### PARSING DATA ###
            log = list(map(int, time_stamp.split())) + [centroidX,
                    centroidY,
                    tailX,
                    tailY,
                    headX,
                    headY]
            writer.writerow(log)
            ### PARSING DATA END ###

        ### EXEC TIME CALC ###
        print("--- %s Total seconds ---" % (time.time() - start_time))
        ###           ###

    ### STOP ###
    if args.nopi is not None:
        stream.release()
    else:
        stream.stop()

elif mode == 'preview':
    print("PREVIEW MODE")
    # drop_frame is used in the drawing section
    drop_frame = False
    for i in range(1000):
            
        ### EXEC TIME CALC ### 
        start_time = time.time()
        ###                ###

        # read a single frame
        if resize is True:
            if args.nopi is not None:
                ret, frame = cv2.resize(stream.read(), (width, height), interpolation = cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.resize(stream.read(), (width, height), interpolation = cv2.INTER_LINEAR)
        else:
            if args.nopi is not None:
                ret, frame = stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = stream.read()


        ### IMAGE PROCESSING ###
        frame_filter = preprocess_image(frame, d, sigma1, sigma2)
        frame_diff = bgfg_diff(bg, frame_filter)
        contours, area, n_contours = contour_extraction(frame_diff)
        frame_post = postprocess_image(contours, kx, ky)
        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        ang = get_orientation(frame_post, frame)
        ### IMAGE PROCESSING END ###

        ### POINTS EXTRACTION ###
        if resize is True:
            area_ratio = area / (width * height)
            print(area_ratio)
        else:
            area_ratio = area / (frame.shape[0] * frame.shape[1])
            print(area_ratio)
        """
            Will only calculate points if:
            The number of contours detected is not 0 and
            The ratio of the largest contour is not greater than
            10% of whole image area
            Otherwise will write 'NA' and drop the frame from further processing
        """
        if n_contours != 0 and area_ratio < 0.1:
            M = cv2.moments(frame_post)
            centroidX = int(M['m10'] / M['m00'])
            centroidY = int(M['m01'] / M['m00'])
            tailX = 0
            tailY = 0
            headX = 0
            headY = 0
            drop_frame = False
        else:
            centroidX = 'NA'
            centroidY = 'NA'
            tailX = 'NA'
            tailY = 'NA'
            headX = 'NA'
            headY = 'NA'
            drop_frame = True
        ### POINTS EXTRACTION END ###
        print("contour number: " + str(n_contours))

        ### DRAWINGS ###
        if drop_frame is False:
            img_jpg = cv2.circle(frame,
                    (centroidX, centroidY),
                    radius=3,
                    color=0,
                    thickness=-1)
            #cv2.imwrite(stream_folder + '/pic{:>05}.jpg'.format(i), img_jpg) 
        else:
            print("Dropped frame")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ### DRAWINGS END ###

        ### EXEC TIME CALC ###
        print("--- %s Total seconds ---" % (time.time() - start_time))
        ###           ###

elif mode == 'offline':
    print("OFFLINE MODE")
    for i in range(len(img_files)):
        start_time = time.time()
        # read a single frame
        frame = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(img_files[i])
        if resize is True:
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
            color = cv2.resize(color, (width, height), interpolation = cv2.INTER_LINEAR)


        ### IMAGE PROCESSING ###
        frame_diff = bgfg_diff(bg, frame)
        frame_post, tail_image = postprocess_image(frame_diff, kx, ky)

        # Contours extractions #
        extraction_body, extraction_tail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err = contour_extraction(frame_post, tail_image, width, height)
        ######################

        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        ### IMAGE PROCESSING END ###

        ### EXEC TIME CALC ###
        print("--- %s Total seconds ---" % (time.time() - start_time))
        ###           ###

        ### DRAWINGS ###
        if err is False:
            # draw body centroid
            cv2.circle(color,
                    (centroidX, centroidY),
                    radius=1,
                    color=(255,0,0),
                    thickness=-1)
            # draw tail centroid
            cv2.circle(color,
                    (centroidXT, centroidYT),
                    radius=1,
                    color=(0,255,0),
                    thickness=-1)
            # draw head estimation
            cv2.circle(color,
                    (centroidXH, centroidYH),
                    radius=1,
                    color=(0,0,255),
                    thickness=-1)
            # draw the whole frame
            cv2.imwrite(stream_folder + '/' + 'frame{:>05}.png'.format(i), color)
        ### DRAWINGS END ###
