#!/bin/bash


#initialization

ssh analyzr7 python XYZ_execute.py 3PA-50
sleep 2
ssh analyzr7 python XYZ_execute.py 1PA-13
sleep 2
ssh analyzr7 python XYZ_execute.py 2PA-31

#___________________________FIRST FLOOOR___________________________________________

for k in $(seq 1 4); 
do
    ssh analyzr7 python XYZ_execute.py 3PR2
    for j in $(seq 1 120)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py') 
        echo "trace $j"
        python3 Pico.py $k $j $lightout $lightin $temperature
    done
    ssh analyzr7 python XYZ_execute.py 3PR-2
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 3
done

ssh analyzr7 python XYZ_execute.py 3PR2
   for j in $(seq 1 120)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py $k $j $lightout $lightin $temperature
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
    ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py $k $j $lightout $lightin $temperature
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done

ssh analyzr7 python XYZ_execute.py 3PR10

   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py 10 $j $lightout $lightin $temperature 
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
    ssh analyzr7 python XYZ_execute.py 3PR9
   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py $k $j $lightout $lightin $temperature 
    
    done
    ssh analyzr7 python XYZ_execute.py 3PR-9
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done

ssh analyzr7 python XYZ_execute.py 3PR9
   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py 15 $j $lightout $lightin $temperature 
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
    ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py $k $j $lightout $lightin $temperature 
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR15
    sleep 2
done

ssh analyzr7 python XYZ_execute.py 3PR10
   for j in $(seq 1 60)
    do
        temperature=$(ssh ir_pi 'python3 /home/irpi/temp.py')
        lightout=$(ssh ir_pi 'python3 /home/irpi/light_outside.py')
        lightin=$(ssh ir_pi 'python3 /home/irpi/light_inside.py')
        echo "trace $j"
        python3 Pico.py 20 $j $lightout $lightin $temperature 
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
