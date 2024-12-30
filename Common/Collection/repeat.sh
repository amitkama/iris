#!/bin/bash


#initialization

ssh analyzr7 python XYZ_execute.py 3PA-50
ssh analyzr7 python XYZ_execute.py 1PA-13
ssh analyzr7 python XYZ_execute.py 2PA-30

#___________________________FIRST FLOOOR___________________________________________

for k in $(seq 1 4); 
do
    ssh analyzr7 python XYZ_execute.py 3PR2
    for j in $(seq 1 110)
    do
        echo "trace $j"
        python3 -u Pico.py $k $j
    done
    ssh analyzr7 python XYZ_execute.py 3PR-2
    ssh analyzr7 python XYZ_execute.py 1PR15
done

ssh analyzr7 python XYZ_execute.py 3PR2
for j in $(seq 1 110)
do
    echo "trace $j"
    python3 -u Pico.py 5 $j
done
ssh analyzr7 python XYZ_execute.py 3PR-2

#___________________________SECOND FLOOOR___________________________________________

ssh analyzr7 python XYZ_execute.py 2PR18

ssh analyzr7 python XYZ_execute.py 1PR-60

ssh analyzr7 python XYZ_execute.py 3PR25

for k in $(seq 6 9);
do
    ssh analyzr7 python XYZ_execute.py 3PR10
    for j in $(seq 1 110)
    do
        echo "trace $j"
        python3 -u Pico.py $k $j
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    ssh analyzr7 python XYZ_execute.py 1PR15
done

ssh analyzr7 python XYZ_execute.py 3PR10
for j in $(seq 1 110)
do
    echo "trace $j"
    python3 -u Pico.py 10 $j
done
ssh analyzr7 python XYZ_execute.py 3PR-10
ssh analyzr7 python XYZ_execute.py 1PR-60

#___________________________THIRD FLOOOR___________________________________________


ssh analyzr7 python XYZ_execute.py 2PR16
ssh analyzr7 python XYZ_execute.py 3PR34

for k in $(seq 11 14);
do
    ssh analyzr7 python XYZ_execute.py 3PR10
    for j in $(seq 1 110)
    do
        echo "trace $j"
        python3 -u Pico.py $k $j
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    ssh analyzr7 python XYZ_execute.py 1PR15
done

ssh analyzr7 python XYZ_execute.py 3PR10
for j in $(seq 1 110)
do
    echo "trace $j"
    python3 -u Pico.py 15 $j
done
ssh analyzr7 python XYZ_execute.py 3PR-10
ssh analyzr7 python XYZ_execute.py 1PR-60


#___________________________FOURTH FLOOOR___________________________________________


ssh analyzr7 python XYZ_execute.py 2PR16
ssh analyzr7 python XYZ_execute.py 3PR31

for k in $(seq 16 19);
do
    ssh analyzr7 python XYZ_execute.py 3PR10
    for j in $(seq 1 110)
    do
        echo "trace $j"
        python3 -u Pico.py $k $j
    done
    ssh analyzr7 python XYZ_execute.py 3PR-10
    ssh analyzr7 python XYZ_execute.py 1PR15
done

# for k in $(seq 16 19)
# do
#     ssh analyzr7 python XYZ_execute.py 3PR10
    
#     if [ "$k" -eq 16 ]; then
#         for j in $(seq 1 110)
#         do
#             echo "trace $j"
#             python3 -u Pico.py $k $j
#         done
#     fi

#     ssh analyzr7 python XYZ_execute.py 3PR-10
#     ssh analyzr7 python XYZ_execute.py 1PR15
# done

ssh analyzr7 python XYZ_execute.py 3PR10
for j in $(seq 1 110)
do
    echo "trace $j"
    python3 -u Pico.py 20 $j
done
ssh analyzr7 python XYZ_execute.py 3PR-10
ssh analyzr7 python XYZ_execute.py 1PR-60
ssh analyzr7 python XYZ_execute.py 3PA-50
sleep 3
ssh analyzr7 python XYZ_execute.py 2PA-30
ssh analyzr7 python XYZ_execute.py 1PA-13
