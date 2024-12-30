#!/bin/bash


#initialization

# ssh analyzr7 python XYZ_execute.py 3PA-50
# sleep 2
# ssh analyzr7 python XYZ_execute.py 1PA-13
# sleep 2
# ssh analyzr7 python XYZ_execute.py 2PA-31

sample_interval=50  # 50 microseconds per sample

#___________________________FIRST FLOOOR___________________________________________

for k in $(seq 2 4); 
do

    directory="Receiver${k}"
    mkdir -p "$directory"

    ssh analyzr7 python XYZ_execute.py 3PR2
    for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py $k $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
    
    # Continue with the rest of the steps
    ssh analyzr7 python XYZ_execute.py 3PR-2
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 3
done


directory="Receiver5"
mkdir -p "$directory"

ssh analyzr7 python XYZ_execute.py 3PR2
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py 5 $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
ssh analyzr7 python XYZ_execute.py 3PR-2
sleep 2

#___________________________SECOND FLOOOR___________________________________________

ssh analyzr7 python XYZ_execute.py 2PR18
sleep 2

ssh analyzr7 python XYZ_execute.py 1PR-60
sleep 2
ssh analyzr7 python XYZ_execute.py 3PR25
sleep 2

for k in $(seq 6 9);
do
    directory="Receiver${k}"
    mkdir -p "$directory"
    ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py $k $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done


directory="Receiver10"
mkdir -p "$directory"
ssh analyzr7 python XYZ_execute.py 3PR10

   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py 10 $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
ssh analyzr7 python XYZ_execute.py 3PR-10
sleep 2


#___________________________THIRD FLOOOR___________________________________________


ssh analyzr7 python XYZ_execute.py 2PR16
sleep 2
ssh analyzr7 python XYZ_execute.py 1PR-60
sleep 2
ssh analyzr7 python XYZ_execute.py 3PR34
sleep 2

for k in $(seq 11 14);
do
    directory="Receiver${k}"
    mkdir -p "$directory"
    ssh analyzr7 python XYZ_execute.py 3PR9
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py $k $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
    ssh analyzr7 python XYZ_execute.py 3PR-9
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done

directory="Receiver15"
mkdir -p "$directory"
ssh analyzr7 python XYZ_execute.py 3PR9
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py 15 $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
ssh analyzr7 python XYZ_execute.py 3PR-9
sleep 2


#___________________________FOURTH FLOOOR___________________________________________


ssh analyzr7 python XYZ_execute.py 2PR16
sleep 2
ssh analyzr7 python XYZ_execute.py 1PR-60
sleep 2
ssh analyzr7 python XYZ_execute.py 3PR31
sleep 2

for k in $(seq 16 19);
do
    directory="Receiver${k}"
    mkdir -p "$directory"
    ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py $k $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done


directory="Receiver20"
mkdir -p "$directory"
ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 120)
    do

        csv_filename="${directory}/trace_num${j}.csv"
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        echo "trace $j"
        
        # Run Pico and wait for it in the background
        python3 Pico.py 20 $j $lightout $temperature &
        sleep 1
        pico_pid=$!  # Capture the PID of the Pico.py process
        
        # Get the analog data from the Raspberry Pi (both channels)
        data=$(ssh ir_pi "python3 /home/irpi/analog_read_8.py")
        wait $pico_pid
        
        # Convert the data to an array (split by commas)
        IFS=',' read -r -a data_array <<< "$data"

        # Write headers to the CSV file
        echo "Time (us),Channel (V)" > "$csv_filename"

        # Initialize time (us) and sample interval (50 us)
        time=0
        sample_interval=50

        # Write the entire data array to the CSV
        for value in "${data_array[@]}"; do
            echo "$time,$value" >> "$csv_filename"
            time=$((time + sample_interval))  # Increment the time by 50 microseconds
        done
    done

ssh analyzr7 python XYZ_execute.py 3PR-10
sleep 3

ssh analyzr7 python XYZ_execute.py 3PA-50
sleep 3
ssh analyzr7 python XYZ_execute.py 1PR-60
sleep 3
ssh analyzr7 python XYZ_execute.py 2PA-30
sleep 3
ssh analyzr7 python XYZ_execute.py 1PA-13
