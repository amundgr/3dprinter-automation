import cv2
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, argrelextrema, find_peaks
import threading

class Printer:
    def __init__(self, serial_port="/dev/ttyUSB1", baudrate=115200, verbose=False):
        self.verbose = verbose
        self.serial_port = serial_port
        self.baudrate = baudrate
        print("pre serial")
        self.ser = serial.Serial(serial_port, baudrate)
        print("post serial")
        self.initiated = False
        while not self.initiated:
            line = self.ser.readline()
            #print(line)
            if b'echo:SD card ok\n' == line:
                self.initiated = True
        print("All good, waiting for position")
        self.get_position()

    def send_command_and_wait(self, command, wait_command="M400"):
        if self.verbose:
            print("\n-----------------------------------")
        self.send_command(command)
        self.send_command(wait_command)

    def send_command(self, command):
        if self.verbose:
            print("Sendt: " + command)
        self.ser.write(f"{command}\r\n".encode())
        ok = False
        while not ok:
            recived = self.ser.readline()
            if self.verbose:
                print(recived)
            ok = "ok" in recived.decode()
            if self.verbose:
                print(recived.decode().strip())
        
    def get_position(self):
        self.ser.write("M114\r\n".encode())
        recived = False
        while not recived:
            rec = self.ser.readline().decode().strip()
            if rec[0] == "X":
                rec = rec.split(" ")
                self.x = float(rec[0].split(":")[-1])
                self.y = float(rec[1].split(":")[-1])
                self.z = float(rec[2].split(":")[-1])
                recived = True
        

    def home(self, x=False, y=False, z=False):
        x = "X" if x else ""
        y = "Y" if y else ""
        z = "Z" if z else ""
        command = f"G28 {x}{y}{z}"
        self.send_command_and_wait(command)
        self.get_position()

    def move_to(self, x=0, y=0, z=0):
        x = f"X{x} " if x else ""
        y = f"Y{y} " if y else ""
        z = f"Z{z}"  if z else ""
        command = f"G1 {x}{y}{z}"
        self.send_command_and_wait(command)
        self.get_position()

if __name__ == "__main__":
    p = Printer()
    p.send_command_and_wait("M206 Z+0.2")