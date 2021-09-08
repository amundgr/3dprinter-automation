import RPi.GPIO as GPIO

class Actuator:
    half_steps = [
        [0,1,1,1],
        [0,0,1,1],
        [1,0,1,1],
        [1,0,0,1],
        [1,1,0,1],
        [1,1,0,0],
        [1,1,1,0],
        [0,1,1,0],
    ]

    increment_value = 0.001

    def __init__(self, pins=[29,31,33,35], zero_pin=37):
        self.pins = pins
        self.zero_pin = zero_pin
        self.index = 0
        self.height = 0
        GPIO.setmode(GPIO.BOARD)
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(zero_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def step(self, n_steps):
        direction = n_steps / abs(n_steps)
        for i in range(n_steps):
            if self.index == 8:
                self.index = 0
                self.height += direction * self.increment_value
            
            for j in range(4):
                GPIO.output(self.pins[j], half_steps[self.index][j])
            self.index += direction
            
            if GPIO.input(self.zero_pin):
                self.height = 0
                return

    def home(self):
        self.step(-100000)


if __name__ == "__main__":
    s = Stepper()