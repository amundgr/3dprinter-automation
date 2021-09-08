import serial
import time

class MeasuringGauge():
    def __init__(self, serial_port="/dev/ttyUSB0", baudrate=115200):
        self.serial_port = serial.Serial(port=serial_port, baudrate=baudrate, timeout=3)
        time.sleep(1)

    def read_value(self):
        self.serial_port.flushInput()
        data = self.serial_port.read(3)
        sign = int(data[0])
        sign = -1 if sign else 1
        lsb = int(data[1])
        msb = int(data[2])  
        value = sign * (lsb + msb * 2**8) / 100
        return value


if __name__ == "__main__":
    p = MeasuringGauge()
    print(p.read_value())
