## C-MHAD-PytorchSolution
This is an Pytorch Solution for sensor fusion based continuous action detection

In these 3 folder, are solutions using video only, inertial only and Fusion approach.
All of the 3 solutions using sliding window to find the action of interest, this sliding window takes 3 seconds, and moving in 0.2 second.
The fusion approach is a 2 input network, using the same structure from indivudual modality.

### C-MHAD Dataset
About C-MHAD There exists no dataset in the public domain providing simultaneous video and inertial data in which actions of interest are performed randomly in continuous action streams. This paper provides such a dataset, named C-MHAD, in which video and inertial data streams appear simultaneously in a continuous way. This dataset includes 3-axis acceleration and 3-axis angular velocity signals from a wearable inertial sensor and videoclips from a camera that last for 2 minutes for a total of 8 hours of data. These data are collected at the same time from 12 subjects performing a set of actions of interest randomly among arbitrary actions of non-interest in a continuous manner. The C-MHAD dataset can be downloaded from this link www.utdallas.edu/~kehtar/C-MHAD.html 

C-MHAD consist of 2 applications, corresponding to smart TV gestures application and transition movements application. The 5 actions of interest in the smart TV application are 1)swipe left, 2)swipe right, 3)wave, 4)draw circle clockwise, and 5)draw circle counterclockwise. The 7 actions of interest in the transition movements application are 1)stand to sit, 2)sit to stand, 3)sit to lie, 4)lie to sit, 5)lie to stand, 6)stand to lie, and 7)stand to fall.

### C-MHAD related papers
The following paper provides more details on the C-MHAD dataset:
[1] H. Wei, P. Chopada, and N. Kehtarnavaz, "C-MHAD: Continuous Multimodal Human Action Dataset of Simultaneous Video and Inertial Sensing," under review Sensors, 2020. To detect and recognize actions of interest in continuous actions streams, the following paper provides a solution:
[2] H. Wei, and N. Kehtarnavaz, "Simultaneous Utilization of Inertial and Video Sensing for Action Detection and Recognition in Continuous Action Streams," to appear in IEEE Sensors Journal, 2020. 
