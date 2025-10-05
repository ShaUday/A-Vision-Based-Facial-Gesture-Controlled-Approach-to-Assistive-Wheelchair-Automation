import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from imutils import face_utils
from utils import *
import imutils
import dlib
import cv2
import pyfirmata
import threading
import sys
import os


def resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores the path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Global variables
movement_active = False
mouth_toggle = False
board = None
speed = 0.3  # Default speed (0.0 to 1.0)
fixed_square_center = None  # Store the fixed square position
square_width = 40
square_height = 35

# BTS7960 Module Pin Definitions
RPWM1 = 5  # Motor 1 Right PWM
LPWM1 = 6  # Motor 1 Left PWM
REN1 = 7  # Motor 1 Right Enable
LEN1 = 8  # Motor 1 Left Enable

RPWM2 = 9  # Motor 2 Right PWM
LPWM2 = 10  # Motor 2 Left PWM
REN2 = 11  # Motor 2 Right Enable
LEN2 = 12  # Motor 2 Left Enable


# Motor Control Functions
def stop_motors():
    """Stop both motors completely"""
    if board:
        # Set PWM to 0 for all PWM pins
        board.digital[RPWM1].write(0.0)
        board.digital[LPWM1].write(0.0)
        board.digital[RPWM2].write(0.0)
        board.digital[LPWM2].write(0.0)

        # Disable all enable pins
        board.digital[REN1].write(0)
        board.digital[LEN1].write(0)
        board.digital[REN2].write(0)
        board.digital[LEN2].write(0)


def move_forward():
    """Move both motors forward - wheelchair moves forward"""
    if board:
        stop_motors()

        # Enable all motors first
        board.digital[REN1].write(1)
        board.digital[LEN1].write(1)
        board.digital[REN2].write(1)
        board.digital[LEN2].write(1)

        # Both motors forward: Left PWM active, Right PWM = 0
        # Motor 1 (Left side) - Forward
        board.digital[RPWM1].write(0.0)
        board.digital[LPWM1].write(speed)

        # Motor 2 (Right side) - Forward
        board.digital[RPWM2].write(0.0)
        board.digital[LPWM2].write(speed)


def move_backward():
    """Move both motors backward - wheelchair moves backward"""
    if board:
        stop_motors()

        # Enable all motors first
        board.digital[REN1].write(1)
        board.digital[LEN1].write(1)
        board.digital[REN2].write(1)
        board.digital[LEN2].write(1)

        # Both motors backward: Right PWM active, Left PWM = 0
        # Motor 1 (Left side) - Backward
        board.digital[RPWM1].write(speed)
        board.digital[LPWM1].write(0.0)

        # Motor 2 (Right side) - Backward
        board.digital[RPWM2].write(speed)
        board.digital[LPWM2].write(0.0)


def turn_left():
    """Turn left: Right motor forward, Left motor backward"""
    if board:
        stop_motors()

        # Enable all motors first
        board.digital[REN1].write(1)
        board.digital[LEN1].write(1)
        board.digital[REN2].write(1)
        board.digital[LEN2].write(1)

        # Left motor backward, Right motor forward
        # Motor 1 (Left side) - Backward
        board.digital[RPWM1].write(speed)
        board.digital[LPWM1].write(0.0)

        # Motor 2 (Right side) - Forward
        board.digital[RPWM2].write(0.0)
        board.digital[LPWM2].write(speed)


def turn_right():
    """Turn right: Left motor forward, Right motor backward"""
    if board:
        stop_motors()

        # Enable all motors first
        board.digital[REN1].write(1)
        board.digital[LEN1].write(1)
        board.digital[REN2].write(1)
        board.digital[LEN2].write(1)

        # Left motor forward, Right motor backward
        # Motor 1 (Left side) - Forward
        board.digital[RPWM1].write(0.0)
        board.digital[LPWM1].write(speed)

        # Motor 2 (Right side) - Backward
        board.digital[RPWM2].write(speed)
        board.digital[LPWM2].write(0.0)


# Load Face Detection Models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(resource_path("shape_predictor_68_face_landmarks.dat"))
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def draw_mouth_landmarks(frame, mouth_points):
    """Draw all mouth landmarks with numbers and connections"""
    # Draw individual mouth landmark points
    for i, (x, y) in enumerate(mouth_points):
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Yellow circles
        cv2.putText(frame, str(i), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw mouth outline (outer lip)
    mouth_hull = cv2.convexHull(mouth_points)
    cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

    # Draw inner mouth (points 12-19 are inner mouth landmarks)
    if len(mouth_points) >= 20:  # Make sure we have all mouth points
        inner_mouth = mouth_points[12:20]
        inner_hull = cv2.convexHull(inner_mouth)
        cv2.drawContours(frame, [inner_hull], -1, (255, 0, 0), 1)


def draw_nose_landmarks(frame, nose_points):
    """Draw nose landmarks for reference"""
    for i, (x, y) in enumerate(nose_points):
        cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)  # Magenta circles
        cv2.putText(frame, f"n{i}", (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


# Tkinter GUI Class
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Speed Slider Style
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("Custom.Horizontal.TScale", background="lightblue", troughcolor="gray",
                             sliderthickness=20, sliderrelief="raised", borderwidth=2)

        # COM Port Input
        self.com_port_label = ttk.Label(window, text="Enter COM Port: ")
        self.com_port_label.pack(pady=5)
        self.com_port_entry = tk.Entry(window, bg="lightblue")
        self.com_port_entry.pack(pady=5)
        self.com_port_entry.bind("<Return>", lambda event: self.start_connection_thread())
        self.connect_button = ttk.Button(window, text="Connect", command=self.start_connection_thread)
        self.connect_button.pack(pady=10)

        # Status Label
        self.status_label = ttk.Label(window, text="Status: Not Connected", foreground="red")
        self.status_label.pack(pady=5)

        # Speed Slider UI
        self.speed_frame = ttk.Frame(window)
        self.speed_frame.pack(pady=5)
        self.low_label = ttk.Label(self.speed_frame, text="Low")
        self.low_label.grid(row=0, column=0, padx=5)
        self.high_label = ttk.Label(self.speed_frame, text="High")
        self.high_label.grid(row=0, column=2, padx=5)
        self.speed_slider = ttk.Scale(self.speed_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=200,
                                      command=self.update_speed, style="Custom.Horizontal.TScale")
        self.speed_slider.set(speed)
        self.speed_slider.grid(row=0, column=1, padx=5)
        self.speed_value_var = tk.StringVar(value=f"Speed: {speed:.2f}")
        self.speed_value_label = ttk.Label(self.speed_frame, textvariable=self.speed_value_var)
        self.speed_value_label.grid(row=1, column=0, columnspan=3, pady=5)

        # Video Display
        self.vid = cv2.VideoCapture(1)
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Start Update Loop
        self.delay = 15
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.close_app)
        self.window.mainloop()

    def start_connection_thread(self):
        threading.Thread(target=self.connect_arduino, daemon=True).start()

    def connect_arduino(self):
        global board
        com_port = self.com_port_entry.get().strip()
        if not com_port:
            self.status_label.config(text="Status: Please enter a COM port", foreground="red")
            return
        try:
            self.status_label.config(text="Status: Connecting...", foreground="orange")
            board = pyfirmata.Arduino(com_port)

            # Set pin modes correctly
            for pin in [RPWM1, LPWM1, RPWM2, LPWM2]:
                board.digital[pin].mode = pyfirmata.PWM

            for pin in [REN1, LEN1, REN2, LEN2]:
                board.digital[pin].mode = pyfirmata.OUTPUT

            self.status_label.config(text=f"Status: Connected to {com_port}", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Status: Failed to connect - {str(e)}", foreground="red")

    def update_speed(self, value):
        global speed
        speed = float(value)
        self.speed_value_var.set(f"Speed: {speed:.2f}")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=640, height=480)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            largest_face = None
            largest_area = 0
            for rect in rects:
                area = rect.area()
                if area > largest_area:
                    largest_area = area
                    largest_face = rect

            if largest_face is not None:
                shape = predictor(gray, largest_face)
                shape = face_utils.shape_to_np(shape)
                mouth = shape[mStart:mEnd]
                nose = shape[nStart:nEnd]
                nose_point = (nose[3, 0], nose[3, 1])
                mar = mouth_aspect_ratio(mouth)

                # Draw landmarks for visualization
                draw_mouth_landmarks(frame, mouth)
                draw_nose_landmarks(frame, nose)

                # Display MAR value
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw face rectangle for reference
                (x, y, w, h) = face_utils.rect_to_bb(largest_face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                global movement_active, mouth_toggle, fixed_square_center
                if mouth_aspect_ratio(mouth) > 0.4:
                    if not mouth_toggle:
                        movement_active = not movement_active
                        mouth_toggle = True
                        # Set the fixed square position when mouth opens and movement becomes active
                        if movement_active:
                            fixed_square_center = (nose_point[0], nose_point[1])
                            cv2.putText(frame, "SQUARE LOCKED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                                        2)
                        else:
                            fixed_square_center = None
                            cv2.putText(frame, "SQUARE UNLOCKED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 0), 2)
                else:
                    mouth_toggle = False

                # Show movement status
                if movement_active:
                    cv2.putText(frame, "MOVEMENT ACTIVE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "MOVEMENT INACTIVE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw the control square (either fixed position or current nose position)
                if movement_active and fixed_square_center:
                    # Use the fixed square position
                    square_x, square_y = fixed_square_center
                    cv2.rectangle(frame, (square_x - square_width, square_y - square_height),
                                  (square_x + square_width, square_y + square_height), (0, 255, 0), 3)
                    cv2.putText(frame, "CONTROL ZONE", (square_x - square_width, square_y - square_height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calculate direction based on current nose position relative to fixed square
                    dir = direction(nose_point, (square_x, square_y), square_width, square_height)

                    # Direction mapping: nose relative to fixed square
                    if dir == 'up':
                        cv2.putText(frame, "FORWARD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if board:
                            move_forward()
                    elif dir == 'down':
                        cv2.putText(frame, "BACKWARD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if board:
                            move_backward()
                    elif dir == 'left':
                        cv2.putText(frame, "TURN LEFT", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if board:
                            turn_left()
                    elif dir == 'right':
                        cv2.putText(frame, "TURN RIGHT", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if board:
                            turn_right()
                    elif dir == 'center':
                        cv2.putText(frame, "STOP", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        if board:
                            stop_motors()

                    # Show current nose position relative to fixed square
                    cv2.circle(frame, nose_point, 5, (255, 0, 0), -1)
                    cv2.putText(frame, "NOSE", (nose_point[0] + 10, nose_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                else:
                    # Movement inactive - stop motors
                    if board:
                        stop_motors()

                    # Show current nose position (preview)
                    cv2.circle(frame, nose_point, 3, (128, 128, 128), -1)
                    cv2.putText(frame, "nose", (nose_point[0] + 5, nose_point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def close_app(self):
        if self.vid.isOpened():
            self.vid.release()
        if board:
            board.exit()
        self.window.destroy()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        if board:
            board.exit()


# Run Application
App(tk.Tk(), "SelfBot Wheelchair")
