from itertools import chain
import csv
import tkinter as tk
import paramiko
import cv2
import numpy as np

def remote_command(ip, port, username, password, command):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, port=port, username=username, password=password)
        stdin, stdout, stderr = ssh.exec_command(command)
        print(stdout.readlines())

    finally:
        if ssh:
            ssh.close()

def test_remote_cam(ip, port, username, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, port=port, username=username, password=password)
        stdin, stdout, stderr = ssh.exec_command('raspistill -v -o test.jpg')
        print(stdout.readlines())

        sftp = ssh.open_sftp()
        with sftp.open('/home/pi/test.jpg') as f:
            img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)

        imS = cv2.resize(img, (960, 540))
        cv2.imshow("image", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        if ssh:
            ssh.close()

def read_csv_row(row):
    row = int(row)
    with open('drop.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
    global pi_csv
    pi_csv = data[row]
    print(row)


window = tk.Tk()

pi_csv = []

test_cam_button = tk.Button(
        text="Test CAM",
        width=25,
        height=5,
        bg="black",
        fg="white",
        command= lambda: test_remote_cam(
            pi_csv[0],
            pi_csv[1],
            pi_csv[2],
            pi_csv[3])
        )
test_cam_button.pack()

pi_num = tk.StringVar()
pi_num.set("0")
pi_number = tk.OptionMenu(window, pi_num, "0", "1", "2")
pi_number.pack()

read_csv = tk.Button(
        text="Read config file",
        width=25,
        height=5,
        bg="black",
        fg="white",
        command= lambda: read_csv_row(pi_num.get())
        )
read_csv.pack()





window.mainloop()
