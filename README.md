# CUDACapstoneProject
This is the submission of **Luca Venturi** for the **GPU Specialization Capstone Project** 

## Project Description
My company developed a custom, columnar NoSql DB for analytics; this program is intended to test some fo the basic functionalities of a similar DB, and evaluate the possible speed-up of using a GPU.

The program define an abstract class, **DBEngine**, that defines some expected functions, without implementing them.
There is a class called **DBEngineCPU**, that implements it with a single thread, both as a speed reference and to verify the correctness of the other solutions.
**DBEngineCPU4Threads** implements a DBEngine using 4 CPU threads.
The GPU implementation using nVidia NPP is provided by **DBEngineNPP**.
In addition, there is a class called DBEngineNPPInit, which is used to initialize the NPP libraries.

## Building it and testing it
The program has been developed and tested on WSL, using Ubuntu 18.04.

To build it, you can use the command **make build**.

To run it, you could do it in different ways, for example:
- ./run.sh
- make run
- ./luca

## Example of output
The following is the output after creating 1 000 000 000 of random numbers:
CPU - Time: 12794 ms - Init: 0 ms - Copy: 0 - Count: 4653 - Average: -11448.7 - Min: -2147483648 - Max: 2147483643
CPU 4 - Time: 4052 ms - Init: 0 ms - Copy: 0 - Count: 4653 - Average: -11448.7 - Min: -2147483648 - Max: 2147483643
NPP INIT - Time: 1136 ms - Init: 472 ms - Copy: 544 - Count: 4653 - Average: -11449 - Min: -2147483648 - Max: 2147483643
NPP - Time: 849 ms - Init: 146 ms - Copy: 558 - Count: 4653 - Average: -11449 - Min: -2147483648 - Max: 2147483643
NPP - pinned - Time: 4382 ms - Init: 1353 ms - Copy: 357 - Count: 4653 - Average: -11449 - Min: -2147483648 - Max: 2147483643

This is with 100 000 000 random numbers:
CPU - Time: 1263 ms - Init: 0 ms - Copy: 0 - Count: 469 - Average: -19081.6 - Min: -2147483627 - Max: 2147483615
CPU 4 - Time: 457 ms - Init: 0 ms - Copy: 0 - Count: 469 - Average: -19081.6 - Min: -2147483627 - Max: 2147483615
NPP INIT - Time: 533 ms - Init: 432 ms - Copy: 55 - Count: 469 - Average: -19082 - Min: -2147483627 - Max: 2147483615
NPP - Time: 75 ms - Init: 2 ms - Copy: 65 - Count: 469 - Average: -19082 - Min: -2147483627 - Max: 2147483615
NPP - pinned - Time: 388 ms - Init: 118 ms - Copy: 34 - Count: 469 - Average: -19082 - Min: -2147483627 - Max: 2147483615


## Lessons learned
* The GPU can speed-up these tasks
* NPP can be very valuable for these tasks, and it is much easier than trying to do it with the Runtime API
* Pinned memory has faster transfers, but slower allocation
* Pinned memory has some issue (maybe related to size and access patterns) that potentially make it quite tricky to use
* GPU code is much longer and low level than CPU code