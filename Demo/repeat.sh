# #!/bin/bash

ssh analyzr7 python XYZ_execute.py 3PA-50
ssh analyzr7 python XYZ_execute.py 1PA-13
ssh analyzr7 python XYZ_execute.py 2PA-30

# Main script logic
check_receiver=$1
move_1=$(( ((check_receiver-1)%5) * 15))

# Example of using the input
# echo "You have chosen to check receiver number $check_receiver."
# echo "Starting Sampling"

# First floor

if ((check_receiver > 0 && check_receiver < 6)); then

    ssh analyzr7 python XYZ_execute.py 1PR$move_1
    sleep 3
    ssh analyzr7 python XYZ_execute.py 3PR2
    sleep 1
    python3 -u Pico.py $check_receiver 1
    ssh analyzr7 python XYZ_execute.py 3PR-2

    #Predicting
    
    predicted_class=$(python3 test.py)
    sleep 2

    ssh analyzr7 python XYZ_execute.py 1PR-$move_1

# Second floor

elif ((check_receiver > 5 && check_receiver< 11 )); then
    
    move_2=18
    move_3=25

    #Adjusting to 2nd floor:

    ssh analyzr7 python XYZ_execute.py 2PR$move_2
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 3PR$move_3

    #Sampling part

    ssh analyzr7 python XYZ_execute.py 3PR10
    sleep 1  
    python3 -u Pico.py $check_receiver 1
    ssh analyzr7 python XYZ_execute.py 3PR-10

    #Predicting

    predicted_class=$(python3 test.py)
    sleep 2

    #Return to base
    ssh analyzr7 python XYZ_execute.py 3PR-$move_3
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR-$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 2PR-$move_2

#Third Floor

elif ((check_receiver > 10 && check_receiver< 16 )); then

    move_2=34
    move_3=59

    #Adjusting to 3rd floor:

    ssh analyzr7 python XYZ_execute.py 2PR$move_2
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 3PR$move_3    

    #Sampling part

    ssh analyzr7 python XYZ_execute.py 3PR10
    sleep 1  
    python3 -u Pico.py $check_receiver 1
    ssh analyzr7 python XYZ_execute.py 3PR-10

    #Predicting

    predicted_class=$(python3 test.py)
    sleep 2

    #Return to base
    ssh analyzr7 python XYZ_execute.py 3PR-$move_3
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR-$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 2PR-$move_2

# Fourth Floor
elif ((check_receiver > 15 && check_receiver< 21 )); then
    move_2=50
    move_3=90

    #Adjusting to 4th floor:

    ssh analyzr7 python XYZ_execute.py 2PR$move_2
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 3PR$move_3    

    #Sampling part

    ssh analyzr7 python XYZ_execute.py 3PR10
    sleep 1  
    python3 -u Pico.py $check_receiver 1
    ssh analyzr7 python XYZ_execute.py 3PR-10

    #Predicting

    predicted_class=$(python3 test.py)
    sleep 2

    #Return to base
    ssh analyzr7 python XYZ_execute.py 3PR-$move_3
    sleep 2
    ssh analyzr7 python XYZ_execute.py 1PR-$move_1
    sleep 2
    ssh analyzr7 python XYZ_execute.py 2PR-$move_2

fi

# Print the predicted class
echo $predicted_class

# Return the predicted class as the last output
echo $predicted_class

#echo "Sample ended"