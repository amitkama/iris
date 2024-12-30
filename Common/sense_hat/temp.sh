  temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
  echo "Temperature on Raspberry Pi: ${temperature} °C"