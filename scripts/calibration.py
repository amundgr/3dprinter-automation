# import cv2
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, argrelextrema, find_peaks
from scipy.interpolate import griddata, interp2d
import threading
import json

from MeasuringGauge import MeasuringGauge
from Printer import Printer

class Calibration():
    def __init__(self, verbose=False, area_of_interest=(70, 300), home=True):
        self.area_of_interest = area_of_interest
        self.gauge = MeasuringGauge(serial_port="/dev/ttyUSB1")
        print("connected to the measuring gauge")
        time.sleep(1)
        self.printer = Printer(serial_port="/dev/ttyUSB0", verbose=verbose)
        print("connected to the printer")
        if home:
            self.home()

    def stop(self):
        self.cam.stop_video_stream()

    def home(self):
        self.printer.move_to(z=30)
        self.printer.home(x=True, y=True)
        self.printer.home(z=True)
    
    def get_measurement_at_position(self, x=0, y=0, z=0, z_safty=20, redundancy=5):
        self.printer.move_to(z=z_safty)
        self.printer.move_to(x=x, y=y)
        self.printer.move_to(z=z)
        time.sleep(0.5)
        value = self.gauge.read_value()
        return {"x" : x, "y" : y, "z" : z, "measurement" : value} 

    def get_gird(self, num, z_height):
        coor_list = []
        interval = (self.area_of_interest[1] - self.area_of_interest[0]) / (num-1)
        for i in range(num):
            row = [[i, y, z_height] for y in range(0,num)]
            if i%2:
                row.reverse()
            for point in row:
                coor_list.append(point)
        
        measurements = []
        self.printer.move_to(z=20) #Lift nozzle from the bed
        self.printer.send_command_and_wait("M190 S60") #Set bed temperature to 60 degrees and wait
        for coor in coor_list:
            measurement = self.get_measurement_at_position(x=coor[0]*interval + self.area_of_interest[0], 
                                                           y=coor[1]*interval + self.area_of_interest[0], z=coor[2])
            measurement["idx"] = [coor[0], coor[1]]
            measurements.append(measurement)
            print(json.dumps(measurement, indent=4))

        with open("res.json", "w") as fp:
            json.dump(measurements, fp, indent=4)

    def find_biggest_outlier(self, data,):
        sum_data = sum([data[i]["measurement"] for i in range(len(data))])
        len_data = len(data)
        mean_data = sum_data/len_data
        differences = [None]*len_data
        for i, value in enumerate(data):
            new_mean = (sum_data - value["measurement"]) / (len_data - 1)
            differences[i] = abs(mean_data - new_mean)
        return differences.index(max(differences)), max(differences)


    def verify_results(self, filename, threshold=1):
        with open(filename, "r") as fp:
            data = json.load(fp)
            
        threshold /= len(data)
        while True:
            idx, difference = self.find_biggest_outlier(data)
            print("Diff:",threshold,"Thresh:", difference)
            if difference > threshold:
                x, y, z = data[idx]["x"], data[idx]["y"], data[idx]["z"] 
                new_data = self.get_measurement_at_position(x=x, y=y, z=z, redundancy=5)
                data[idx]["measurement"] = new_data["measurement"]
            else:
                break
                
        
        with open("res_clean.json", "w") as fp:
            json.dump(data, fp)

    def adjust_corners(self, z_height):
        self.get_measurement_at_position(x=self.area_of_interest[0], y=self.area_of_interest[0], z=z_height)
        input()
        self.get_measurement_at_position(x=self.area_of_interest[0], y=self.area_of_interest[1], z=z_height)
        input()
        self.get_measurement_at_position(x=self.area_of_interest[1], y=self.area_of_interest[1], z=z_height)
        input()
        self.get_measurement_at_position(x=self.area_of_interest[1], y=self.area_of_interest[0], z=z_height)
        input()

        return

    def find_gauge_nozzle_offset(self):
        self.printer.move_to(z=20)
        current_height = 20
        self.printer.move_to(x=150, y=150, z=20)
        while True:
            val = input("Distance to lower:")
            if val.lower()=="stop":
                break
            else:
                current_height += float(val)
                print("Moving to:", current_height)
                self.printer.move_to(z=current_height)

        self.printer.move_to(z=20)
        self.printer.move_to(x=150+34, y=150+43, z=20)
        self.printer.move_to(z=current_height+0.1)
        self.printer.move_to(z=current_height)
        print(self.gauge.read_value())

    def write_grid_to_printer(self, mesh_file, x_offset=-34, y_offset=-43, z_offset=-9.27):
        with open(mesh_file, "r") as fp:
            mesh = json.load(fp)
        for point in mesh:
            X = point["x"] + x_offset
            Y = point["y"] + y_offset
            Z = point["z"] + z_offset + point["measurement"]
            command = f"M421 X{X} Y{Y} Z{Z}"
            print(command)
            self.printer.send_command_and_wait(command)
            time.sleep(0.5)
        self.printer.send_command_and_wait("M500")

    def extract_correct_grid(self, filename="res.json", n_points=7, gauge_offset_x=-34, gauge_offset_y=-43, gauge_offset_z=-9.7, z_offset=0, plot=False):
        points = []
        values = []
        with open(filename, "r") as fp:
            data = json.load(fp)

        for point in data:
            x = point["x"] + gauge_offset_x
            y = point["y"] + gauge_offset_y
            points.append([x,y])
            values.append(point["measurement"] + gauge_offset_z + point["z"])

        points = np.array(points)
        values = np.array(values)
        #grid_x, grid_y = np.meshgrid(points[:,0], points[:,1])


        f = nearest_nabor(x=points[:,0], y=points[:,1], z=values)


        x_new = np.linspace(0,300,n_points)
        y_new = np.linspace(0,300,n_points)
        z_new = f(x_new, y_new).transpose() + z_offset
        print(np.min(z_new))
        
        if plot:
            plt.imshow(z_new)
            plt.savefig("new_grid")
        
        for i in range(n_points):
            for j in range(n_points):
                Z = z_new[i,j]
                command = f"M421 I{i} J{j} Z{Z}"
                print(command)
                self.printer.send_command_and_wait(command)
                time.sleep(0.5)
        self.printer.send_command_and_wait("M500")

    def calibrate_gauge(self, start_z=1, stop_z=10, increment=0.02):
        self.printer.move_to(z=20)
        self.printer.move_to(x=150, y=150)
        gauge_heights = []
        printer_heights = []
        for i in np.arange(start_z,stop_z,increment):
            self.printer.move_to(z=i)
            printer_heights.append(i)
            time.sleep(0.5)
            gauge_heights.append(self.gauge.read_value())
        res = {"gauge" : gauge_heights, "printer" : printer_heights}
        with open("gauge_calibration.json", "w") as fp:
            json.dump(res, fp)


def plot_calibration_data(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)

    g = np.array(data["gauge"])
    p = np.array(data["printer"])

    g -= np.min(g)
    p -= np.max(p)
    p = np.abs(p)

    diff = p-g

    plt.plot(g[10:-10], diff[10:-10])
    plt.xlabel("Value on gauge [mm]")
    plt.ylabel("Difference from stepper [mm]")
    plt.savefig("gauge_vs_printer", dpi=500)


def nearest_nabor(x,y,z):
    return interp2d(x, y, z, kind='linear')

def polynomial(x,y,z):
    x_range = np.unique(x)
    y_range = np.unique(y)

    Z = np.zeros((len(x_range), len(y_range)))

    for i, val in enumerate(z):
        idx_x = int(round(x[i] / max(x_range) * len(x_range) - 1))
        idx_y = int(round(y[i] / max(y_range) * len(y_range) - 1))
        Z[idx_x, idx_y] = val

    X, Y = np.meshgrid(x_range,y_range,copy=True)

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    print("coeff:",coeff)
    print("r:",r)
    print("rank:",rank)
    print("s:",s)


# y = -43
# x = -34
# z = 9.27

def plot_grid(filename, outfile="figure", show_numbers=True):
    with open(filename, "r") as fp:
        grid = json.load(fp)

    shape = int(np.sqrt(len(grid)))

    grid_matrix = np.zeros((shape, shape))

    for point in grid:
        x, y = point["idx"]
        grid_matrix[x,y] = point["measurement"]

    bias = np.mean(grid_matrix)
    grid_matrix -= bias

    grid_matrix = np.round(grid_matrix * 100) / 100
    
    fig, ax = plt.subplots()
    im = ax.imshow(grid_matrix, cmap=plt.get_cmap("RdBu"))

    # Loop over data dimensions and create text annotations.
    if show_numbers:
        for i in range(shape):
            for j in range(shape):
                text = ax.text(j, i, grid_matrix[i, j],
                            ha="center", va="center", color="k", fontsize=5)

    ax.set_title("Printer bed")
    fig.tight_layout()
    plt.savefig(outfile)


if __name__ == "__main__":
    # Bed measure area [70, 300]
    c = Calibration(home=True)
    c.adjust_corners(z_height=5)
    c.home()
    # c.calibrate_gauge()
    # plot_calibration_data("gauge_calibration.json")
    #c.find_gauge_nozzle_offset()
    #c.write_grid_to_printer(mesh_file="res.json")
    #c.adjust_corners(z_height=5)
    c.get_gird(num=5, z_height=5)

    #c.verify_results("res.json")
    plot_grid("res.json", outfile="data_points", show_numbers=True)
    #c.extract_correct_grid(plot=True, z_offset=0.25)