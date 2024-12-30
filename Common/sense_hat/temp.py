from sense_hat import SenseHat

# Initialize SenseHat
sense = SenseHat()

def get_temperature():
    # Read the temperature from the Sense HAT temperature sensor
    temperature = sense.get_temperature()

    # Optionally, you can convert the temperature from Celsius to Fahrenheit
    # Uncomment the following lines if you want temperature in Fahrenheit
    # temperature = temperature * 9/5 + 32

    return temperature

def main():
    # Get temperature
    temp = get_temperature()

    # Print the temperature
    print(f"Temperature: {temp} °C")

if __name__ == "__main__":
    main()
