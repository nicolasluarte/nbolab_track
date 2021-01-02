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
parserArg.add_argument('-f', '--folder', type=str, help='folder where the png images are')
parserArg.add_argument('-n', '--previewnumber', type=int, help='how much preview frames')
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
nfs = home + '/nfs' + '/' + str(os.uname()[1])
hostname = os.popen('hostname').read().rstrip('\n')
csv_files = nfs + '/csv'
stream_folder = nfs + '/preview'
backgrounds = nfs + '/background'

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

"""
    Set a variable to control de type of execution
    Experiment: no preview; saves results into csv
    Preview: preview from cam; saves images to shared NFS
    Offline: no preview, saves results into csv uses already saved images
"""
if args.mode is not None:
    mode = args.mode
    print("Selected mode is: " + str(mode))
else:
    mode = 'experiment'
    print("Selected mode is: " + str(mode))


# set resize
if args.width is not None and args.height is not None:
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
if args.mode == 'offline':
    img_files = glob.glob(args.folder + '/*.png')

if args.mode != 'offline':
    print("Foreground size: " + str(test_frame.shape))
elif resize is False:
    test_frame = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
    print("Image size: " + str(test_frame.shape))
    width = test_frame.shape[0]
    height = test_frame.shape[1]
    print("width: " + str(width))
    print("height: " + str(height))
else:
    test_frame = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)

# set the number of preview frames
if args.previewnumber is not None:
    preview_n = args.previewnumber
else:
    preview_n = 100

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

# set the background, uses the background folder as default
if args.background is not None:
    bg = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
    print("User specified background")
    print("Background size:" + str(bg.shape))
else:
    print("Taking background pic")
    path = backgrounds + '/' + 'bg.png'
    if args.nopi is not None:
       ret, background_frame = stream.read()
    else:
       background_frame = stream.read()
    cv2.imwrite(path, background_frame)
    bg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("Loading background")

if resize is True:
    bg = cv2.resize(bg, (width, height), interpolation = cv2.INTER_AREA)
else:
    bg = bg


if mode == 'null':
    exit()

elif mode == 'experiment':
    print("Experiment mode selected")
    with open(csv_files + label + '.csv', 'w') as f:

        ### CSV HEADERS ###
        writer = csv.writer(f)
        writer.writerow([
            "YEAR",
            "MONTH",
            "DAY",
            "HOUR",
            "MINUTE",
            "SECOND",
            "MICROSECOND",
            "body_centroid_x",
            "body_centroid_y",
            "tail_centroid_x",
            "tail_centroid_y",
            "head_centroid_x",
            "head_centroid_y",
            "body_contour_area",
            "error",
            "execution_time",
            "frame_width",
            "frame_height"])
        ### CSV HEADERS END ###

        ### EXEC TIME CALC ### 
        start_time = time.time()
        ###                ###

        while True:

            # read a single frame
            if args.nopi is not None:
                ret, frame = stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # in this case the frame is already 1 channel
                frame = stream.read()

            if resize is True:
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)

            ### IMAGE PROCESSING ###
            # Add preprocessing if needed
            # preprocess_image(...)
            frame_diff = bgfg_diff(bg, frame)
            frame_post, tail_image = postprocess_image(frame_diff, kx, ky)

            # Contours extractions #
            extraction_body, extraction_tail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err = contour_extraction(frame_post, tail_image, width, height)
            ######################
            time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
            ### IMAGE PROCESSING END ###

            ### EXEC TIME CALCL ###
            exec_time = (time.time() - start_time)
            ###                 ###

            ### PARSING DATA ###
            log = list(map(int, time_stamp.split())) + \
                    [
                    centroidX,
                    centroidY,
                    centroidXT,
                    centroidYT,
                    centroidXH,
                    centroidYH,
                    area,
                    err,
                    exec_time,
                    width,
                    height
                    ]
            writer.writerow(log)
            ### PARSING DATA END ###

    ### STOP ###
    if args.nopi is not None:
        stream.release()
    else:
        stream.stop()

elif mode == 'preview':
    print("Preview mode selected")
    print("Deleting frames")
    
    path = stream_folder + '/*.png'
    to_delete = glob.glob(path)
    for f in to_delete:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    # set specs for marking the points
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    lineType = 2
    colour = (0, 0, 255)
    thickness = 2

    for counter in range(preview_n):

        ### EXEC TIME CALC ### 
        start_time = time.time()
        ###                ###

        # read a single frame
        if args.nopi is not None:
            ret, frame = stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = stream.read()

        if resize is True:
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
            color = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
        else:
            color = frame


        ### IMAGE PROCESSING ###
        # Add preprocessing if needed
        # preprocess_image(...)
        frame_diff = bgfg_diff(bg, frame)
        frame_post, tail_image = postprocess_image(frame_diff, kx, ky)

        # Contours extractions #
        extraction_body, extraction_tail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err = contour_extraction(frame_post, tail_image, width, height)
        ######################
        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        ### IMAGE PROCESSING END ###

        ### EXEC TIME CALC ###
        exec_time = (time.time() - start_time)
        ###           ###

        ### DRAWINGS ###
        if err is False:
            # draw body centroid
            cv2.circle(color,
                    (centroidX, centroidY),
                    radius=1,
                    color=(0,0,255),
                    thickness=-1)
            cv2.putText(color,
                    'B',
                    (centroidX, centroidY),
                    font,
                    fontScale,
                    colour,
                    thickness,
                    cv2.LINE_AA)
            # draw tail centroid
            cv2.circle(color,
                    (centroidXT, centroidYT),
                    radius=1,
                    color=(0,0,255),
                    thickness=-1)
            cv2.putText(color,
                    'T',
                    (centroidXT, centroidYT),
                    font,
                    fontScale,
                    colour,
                    thickness,
                    cv2.LINE_AA)
            # draw head estimation
            cv2.circle(color,
                    (centroidXH, centroidYH),
                    radius=1,
                    color=(0,0,255),
                    thickness=-1)
            cv2.putText(color,
                    'T',
                    (centroidXH, centroidYH),
                    font,
                    fontScale,
                    colour,
                    thickness,
                    cv2.LINE_AA)

            # draw the whole frame
            cv2.imwrite(stream_folder + '/' + 'frame{:>05}.png'.format(counter), color)
        ### DRAWINGS END ###


elif mode == 'offline':
    print("Offline mode selected")
    with open(csv_files + label + '.csv', 'w') as f:

        ### CSV HEADERS ###
        writer = csv.writer(f)
        writer.writerow([
            "YEAR",
            "MONTH",
            "DAY",
            "HOUR",
            "MINUTE",
            "SECOND",
            "MICROSECOND",
            "body_centroid_x",
            "body_centroid_y",
            "tail_centroid_x",
            "tail_centroid_y",
            "head_centroid_x",
            "head_centroid_y",
            "body_contour_area",
            "error",
            "execution_time",
            "frame_width",
            "frame_height"])
        ### CSV HEADERS END ###

        # Read images from path
        for i in range(len(img_files)):
            start_time = time.time()
            # read a single frame
            frame = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(img_files[i])
            # resize them if needed
            if resize is True:
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
                color = cv2.resize(color, (width, height), interpolation = cv2.INTER_LINEAR)

            ### IMAGE PROCESSING ###
            # Add preprocessing if needed
            # preprocess_image(...)
            frame_diff = bgfg_diff(bg, frame)
            frame_post, tail_image = postprocess_image(frame_diff, kx, ky)

            # Contours extractions #
            extraction_body, extraction_tail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err = contour_extraction(frame_post, tail_image, width, height)
            ######################
            time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
            ### IMAGE PROCESSING END ###

            ### EXEC TIME CALC ###
            exec_time = (time.time() - start_time)
            ###           ###

            ### PARSING DATA ###
            log = list(map(int, time_stamp.split())) + \
                    [
                    centroidX,
                    centroidY,
                    centroidXT,
                    centroidYT,
                    centroidXH,
                    centroidYH,
                    area,
                    err,
                    exec_time,
                    width,
                    height
                    ]
            writer.writerow(log)
            ### PARSING DATA END ###
