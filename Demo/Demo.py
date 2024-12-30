# ****************************************************************************
# This should be running on the client side, and connect to the server via SSH
# ****************************************************************************

import tkinter as tk
from PIL import Image, ImageTk
import paramiko
import os
from tkmacosx import Button

def run_remote_script(hostname, username, password, script_dir, script_name, param):
    """
    SSH into a host and run a script with a parameter after changing to the script directory.

    :param hostname: str, the hostname of the remote server
    :param username: str, the SSH username
    :param password: str, the SSH password
    :param script_dir: str, the directory where the script is located
    :param script_name: str, the name of the script to run
    :param param: str, the parameter to pass to the script
    :return: tuple, the standard output and standard error of the script execution
    """
    try:
        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote server with password authentication
        ssh_client.connect(hostname=hostname, username=username, password=password)

        # Run the script with the parameter
        command = f'cd {script_dir} && bash {script_name} {param}'
        stdin, stdout, stderr = ssh_client.exec_command(command)

        # Read the output and error
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()

        # Close the connection
        ssh_client.close()

        return output, error

    except Exception as e:
        return None, str(e)

# Set to keep track of green buttons
green_buttons = set()

# Function to handle button click
def button_click(number, buttons):
    print(f"Button {number} clicked")
    hostname = '********'  # Replace with your SSH hostname
    username = '********'  # Replace with your SSH username
    password = '********'  # Replace with your SSH password
    script_dir = '~/IRIS_Repository/Demo'
    script_name = 'repeat.sh'

    # Set only the clicked button to gray
    for btn in buttons.values():
        if btn.cget('bg') != 'green':  # Skip green buttons
            btn.config(bg='white')
    buttons[number].config(bg='gray')
    root.update_idletasks()  # Ensure UI updates immediately
    print(f"Button {number} set to gray")

    # Run the remote script with the button number as parameter
    output, error = run_remote_script(hostname, username, password, script_dir, script_name, str(number))
    print("Output:", output)
    print("Error:", error)

    if output:
        # Capture the ML output from the output of repeat.sh
        try:
            ml_output = int(output.split()[-1])
            print("ML Output:", ml_output)
            challenge = ((ml_output - 1) // 20) + 1
            receiver = ((ml_output - 1) % 20) + 1

            # Color buttons based on the ML output
            if number == receiver:
                buttons[number].config(bg='green')  # Color the button green if it matches the ML output
                green_buttons.add(number)
                root.update_idletasks()  # Ensure UI updates immediately
                print(f"Button {number} set to green")
            else:
                buttons[receiver].config(bg='red')  # Color the ML output button red if it doesn't match the clicked button
                root.update_idletasks()  # Ensure UI updates immediately
                print(f"Button {receiver} set to red")
        except ValueError:
            print("Error parsing ML output")

        # Speak the ML output
        speak(f"The receiver is, {receiver}, Challenge, {challenge}")

        # Reset button colors after 5 seconds, excluding green buttons
        root.after(5000, lambda: reset_button_colors(buttons))

# Function to reset button colors to white, excluding green buttons
def reset_button_colors(buttons):
    for btn_number, btn in buttons.items():
        if btn_number not in green_buttons:
            btn.config(bg='white')

def speak(text, voice='Daniel'):
    os.system(f'say -v {voice} "{text}"')

# Create the main window
root = tk.Tk()
root.title("GUI with Buttons on Image")

# Load and resize the image
image_path = "GUI.png"  # Replace with the path to your image
desired_width = 700  # Set your desired width
desired_height = 700  # Set your desired height

image = Image.open(image_path)
image = image.resize((desired_width, desired_height), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)

# Create a canvas and put the image on it
canvas = tk.Canvas(root, width=desired_width, height=desired_height)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=photo)

# Button coordinates
button_coords = [
    (159, 533), (236, 533), (314, 533), (392, 533), (470, 533),
    (159, 396), (236, 396), (314, 396), (392, 396), (470, 396),
    (159, 251), (236, 251), (314, 251), (392, 251), (470, 251),
    (159, 111), (236, 111), (314, 111), (392, 111), (470, 111)
]

# Dictionary to store buttons
buttons = {}

# Function to place buttons
def place_buttons():
    for i, (x, y) in enumerate(button_coords):
        btn_number = i + 1
        btn = Button(root, text=f"{btn_number}", command=lambda num=btn_number: button_click(num, buttons),
                     width=60, height=60, bg='white', fg='black', font=("Helvetica", 30))
        canvas.create_window(x, y, anchor=tk.NW, window=btn)
        buttons[btn_number] = btn  # Store the button in the dictionary

# Place the buttons on the image
place_buttons()

# Start the Tkinter main loop
root.mainloop()