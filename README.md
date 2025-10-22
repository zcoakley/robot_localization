# Robot Localization Using a Particle Filter

**Zahra Lari, Lily Wei, Zara Coakley**

## Project Goal

The goal of this project was to learn about the particle filter, a fundamental concept in robot localization. A particle filter is an algorithm that determines the location of a robot within a known map by maintaining a set of guesses of where the robot is (the guesses are known as particles). It prunes this set of guesses by comparing the robot's LIDAR data to what each particle "sees" based on its location within the known map. We implemented a working particle filter in ROS that works with a Neato robot.

## Key Steps of a Particle Filter

Our particle filter follows this set of steps:

**Initialization**  
Generate a set of particles representing possible robot positions. These particles are centered around an initial guess that can be set when the program is started.

**Prediction Step**  
Move each particle based on the action the robot takes, using data from the Neato's odometry. Some amount of noise is added because the odometry data is not perfect and the actual movement of the Neato may be slightly different from what was measured.

**Correction Step**  
Laser scan data is used to assign weights to each particle based on how closely the predicted particle matches the robot's actual sensor data. To save computation, we use a pre-calculated table of distances to the nearest obstacle from many points on the map to aid in our calculation.

**Pose Estimation**  
The robot's current position is estimated based on the particle with the highest weight.

**Resampling**  
Particles are resampled from the old particles, favoring particles with higher weights.

**Repeat**  
Steps 2 to 5 get repeated continuously, allowing the point cloud to converge on the robot's actual position.

## A Key Design Decision

A key design decision we made was our strategy for converting between frames when we updated the particle positions using odometry data. The odometry data was in the form <Δx, Δy, Δtheta>. While the odometry data was in the odom frame (a frame initialized based on the Neato's position when the program starts), the particle positions are stored in the world frame, meaning we couldn't just directly apply the change in position from the Neato's odometry to the particles (we were able to directly apply the change in heading however, since the world and odom frames both share the same "heading" axis). However, since we don't actually know where the Neato is or where it started, we can't directly convert from the odometry frame to the world frame either. To solve this, we thought about the problem slightly differently. Instead of trying to directly apply the Δx and Δy, which would require us to know the transform between the odometry and world frames, we instead thought of the change in position in terms of a rotation and then a movement forward. Essentially, we rotated the particle to face the spot where it should end up, then move the correct distance to get there. You can also think of this as a vector, which has a heading facing where the particle should go and a magnitude indicating how far it should move. This vector can be shown in blue in the figure below.

This technique worked because the movement vector is only defined based on the position of the Neato itself and not the odometry frame, meaning we could just apply the same changes (shift the angle then move forward) to the particle without worrying about the odometry frame. To move each particle this way, we defined a vector in the world frame with magnitude d (the distance the particle needs to move, which can be found from Δx and Δy using the pythagorean theorem). We then rotated it to face the same way as the particle, and rotated it again an amount equal to theta_turn (from the figure above) so it would face in the direction it needed to move. This movement vector now indicates how the particle needs to move in the world frame and can be added to the existing position values of the particle.

## Challenges

The main challenge we faced was nailing down our high level understanding of the particle filter. We often took breaks in the middle of pair programming to white board out and discuss what we thought each step of the particle filter should do. One of our biggest points of confusion was with `update_particles_with_odom()` and `update_particles_with_laser_scan()`. We weren't sure exactly what the odometry step should be doing and how it played into the laser scan step. We also spent a lot of time discussing how the odometry step could meaningfully update the particles as we do not know what the initial orientation of the robot is. Simply adding the action the robot takes to the particles didn't seem reliable. In the end, we came to the understanding that the purpose of the odometry step is to give a rough prediction. The laser scan step is responsible for validating the particles based on the sensor data.

Last project, we spent a lot more time in the beginning working and did a good job of spacing deadlines out. This time a lot more of our work time was focused at the end of the project.

## Future Improvements

- Instead of initializing our particles around a guess, we could initialize the particles randomly instead. This would make more sense for applications where the robot has no idea where it is and has to figure it out from its surroundings.
- We could also try adding more uncertainty into our data to see how long the particle filter holds up. We could make our lidar measurements more noisy, or we could make the map less accurate or make it have some areas missing. This would give us an idea of the robustness of our particle filter and we could experiment with how to make it more reliable.
- It would also be fun to learn about how a particle filter could fit into a simple implementation of SLAM. We could experiment with mapping strategies to go along with our particle filter.
- Finally, we could test the computational efficiency of our particle filter by recording how long a single run of the filter takes. We could then experiment with ways to make the computation faster (like writing it in C++!).

## Lessons Learned

We found this project harder to split up work for than the first one, especially as a team of three. Since all parts of the code worked together, it wasn't really possible to just split up the work and separately work on things. We really had to coordinate a lot to make sure we were using the same version of the code and that we weren't stepping on each other's toes too much by writing the same functions separately in two different ways. However, it was a good experience working closely together to understand the concepts. We found it helpful to discuss the conceptual material as a group, walking through how the code should work, and it was useful to have all three of us to discuss and offer up insights.

We found that visualizing what our code is doing in Rviz is essential for verifying what it's doing. You can do a little bit with print statements, but it's pretty much impossible to interpret what the numbers mean without a visualization, so relying on Rviz was very helpful. Another team also gave us the idea to use gazebo to control the Neato manually for debugging, which was super helpful as well, since we knew exactly
