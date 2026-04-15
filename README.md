| Deliverable | Due Date              |
|---------------|----------------------------------------------------------------------------|
| Race Day | Saturday, May 9th 9AM - 4PM EST |
| Code Pushed to Github  | Saturday, May 9th 1PM EST |
| Briefing Slides | Monday, May 4th at 1PM EST |
| Briefing (15 min presentation + 5 min Q&A) | May 4th or 6th at 3PM EST |
| [Team Member Assessment](https://forms.gle/FuwuAduxHc7Q8iQz6)  | Saturday, May 9th at 11:59PM EST |

## Table of Contents

* [Introduction](https://github.com/mit-rss/final_challenge2026#introduction)
    * [Grading](https://github.com/mit-rss/final_challenge2026#grading)
* [The Great Snail Race](https://github.com/mit-rss/final_challenge2026#parta)
* [Mrs. Puff Boating School](https://github.com/mit-rss/final_challenge2026#partb)
* [Briefing](https://github.com/mit-rss/final_challenge2026#Briefing)
* [General Notes](https://github.com/mit-rss/final_challenge2026#general_notes)
* [FAQ](https://github.com/mit-rss/final_challenge2026#faq)

# Final Challenge 2026

### Introduction

Congratulations on completing the six labs of RSS! 

This semester, you've learned how to implement real-time robotics software on a widely-used software framework (ROS2). You know how to read sensor data (LiDAR, Camera, Odometry) and convert it into a useful representation of the world (homography, localization). You've written algorithms that make plans over the world's state (parking, line following, path planning) and combined them with controllers (PD control, pure pursuit) to accomplish tasks. 

Now, your team will apply everything you've learned to successfully pass your boating test and race your snail (i.e your racecar)!

<img src="media/patty_wagon.png" width="500"/>

### Spongebob…!

You have been perfecting your racecar for the last three months. Now, it's time to put your boating skills to the test! Each team will step into the role of SpongeBob, and you have two key
missions: prove your status as the best Gary owner, and pass Mrs. Puff’s boating test and finally earn your license. There will be two parts to this challenge —- first, becoming the fastest team to complete the Bikini Bottom Snail Race, and second, safely navigating through Bikini Bottom while following the rules of the road.
  - In The Great Snail Race, your team will go head-to-head with other teams to be the fastest snail without shimmying out of your lane.
  - In Mrs. Puff's Boating School, you will need to navigate safely through Bikini Bottom, obeying all laws of the road without hitting any unsuspecting fishfolk
    
Luckily, through RSS, you’ve learned everything you need to become the best sponge under the sea!

### Grading

| Deliverable  Grade | Weighting             |
|---------------|----------------------------------------------------------------------------|
| Part A: The Great Snail Race (out of 10) | 35% |
| Part B: Mrs. Puff's Boating School (out of 10) | 25% |
| Briefing Grade (out of 10) | 40% |

## Part A: The Great Snail Race <a name="parta"></a>

<img src="media/snail_race.jpeg" width="400"/>

<img src="media/snail_race_2.jpg" width="400"/>

### Environment and Task

The crowd is roaring, the bubbles are floating, and it’s time for the most prestigious event in all of Bikini Bottom… the Great Snail Race! It’s all about speed, slime, and snail pride. SpongeBob SquarePants arrives with his beloved snail, Gary. Squidward Tennisballs (or is it Tentacles) enters Snelly, the high-pedigree purebred snail. And lastly, Patrick Star brings Rocky… the rock.

The race will take place on the legendary Bikni Botton Johnson Track Loop—a standard 200-meter course. Each snail (that’s your car!) will be assigned to one of six lanes, revealed on race day. Stay in your lane. No snail likes a lane hog. Lanes are numbered from left to right as shown in the image below. Pick your favorite character and prepare for glory. Will you channel SpongeBob’s unstoppable enthusiasm, Squidward’s questionable confidence, or Patrick’s… unwavering belief in a rock?

<!-- <img src="media/final_race.PNG" width="300" /> -->
<img src="media/start_area.jpg" width="400"/>

Your car's task is to complete the 200-meter loop around the track as fast as possible, while staying in your assigned lane. Any kind of collision (with another car, snail, or with something in Johnson) will seriously jeapordize your racer score, and will be penalized heavily. You should have a safety controller running on your car, but be careful that this doesn't stop your car if there is another car driving next to it on the track!

We have provided images and rosbags of the race track in `/racetrack_images` for easier testing/debugging. 

The rosbag can be downloaded at this [link](https://drive.google.com/file/d/1laqouQzSVUhgAsqJVQ08WQdYrOwvVxg0/view?usp=sharing)

Part A is worth 35% of your Final Challenge technical grade. Your grade will be calculated based on the time your car takes to drive around the track (`best_race_split`, in seconds) as follows:

  `Part A grade = min(10 + (5 - best_race_split/10), 11) - penalties`

### Scoring

`best_race_split` will be the fastest of your three runs in seconds.

**Formula for Penalties**

Where `penalties` is calculated as follows:

  `penalties = 1.5 * num_collisions + 0.5 * num_lane_line_breaches + 0.5 * num_long_breaches`
  
`num_lane_line_breaches` is the number of times the car drives outside of either lane line, and `num_long_breaches` is the number of times the car has driven outside of its lane and stayed outside of the lane for greater than 3 seconds.

As you can see from this grading scheme, it is possible to receive bonus points for a very fast and precise solution. The **maximum speed of your car should be capped at 4 m/s**; you should be able to get full points (with bonus!) with a good controller. You should, above all, prioritize avoiding collisions, and if your car leaves its lane, it should quickly recover. More information about race day can be found below in this handout.

### Race Day!
On race day, multiple teams will set up on the track at the same time. A TA will give the start signal, at which point the race begins! You must have a member of your team closely follow your car along the track with the controller ready to take manual control at any moment (yes, a great opportunity to exercise). Your car's split will be recorded at the finish line, and TAs will also be stationed at various points along the track recording lane breaches, if they occur (but hopefully no collisions). Each team will have the opportunity to race **three** times, and we will take the best score.

### Tips

Here are some things you may consider in developing your approach:

- How can you reliably segment the lane lines?
- How can you obtain information about the lane lines in the world frame?
- How can you detect if the car has drifted into a neighboring lane?

Please note that Hough Transforms will very likely be useful; helpful resources are [here](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549) and [here](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html).

## Part B: Mrs. Puff's Boating School <a name="partb"></a>

<img src="media/boating_school.png" width="500"/>

Part B is worth 25% of your Final Challenge technical grade. You get 3 attempts and your grade is based on your best attempt out of 3. Your grade will be calculated based on completion of the course and the number of penalties you incur as follows. 

`Part B grade = boating_test_score - penalties`

### Environment and Task

Mrs. Puff's Boating Test will take place in Bikini Bottom (Stata basement). You, as Spongebob, need to pass your [1,258,057th attempt](http://en.spongepedia.org/index.php?title=Mrs._Puff,_You%E2%80%99re_Fired_(Episode)) at the boating test and get your license (no loopholes allowed)! This will require you to successfully navigate through the city, avoid pedestrians, and correctly stop for traffic lights and parking meters. 

Your goal, after finishing the race successfully, is to drive through the course in Stata basement, where you will be asked to park at 2 TA selected locations. At each location, there will be three roadisde objects: you must correctly identify the parking meter, and park in front of it before continuing on your way. Along the way, there will be a traffic light and pedestrian crosswalk as well. The exact configuration of locations and driving signs will be a secret until Test day; however, the location of the pedestrian crosswalk and traffic light will not change. 

Here are the details of the challenge:

* You will be given 2 locations at random via the `basement_point_publisher` node -- see the Stata map for an example of possible locations. We will not assign locations outside of the range of the ones on the map. 
* You must detect the correct sign out of the three objects at each location using YOLO and park in front of it (stop for 5 seconds)
* You should avoid running the traffic light, taking out pedestrians, or otherwise creating havoc on the road.
* You should drive Mrs. Puff back to the starting location if possible; this might earn you extra points in her book!

***!!!UPDATED!!!*** Things to note: 
* Successful parking entails stopping within ***1m*** in front of the correct item and saving the image with the bounding box
   * If you are having trouble doing both, prioritize ***saving the image***.
* The signs will be ***propped up on the ground*** or **taped along the bottom of the wall** (so homography and YOLO should both work)
* You should stop once you see the red light, but you can go once you stop seeing the red light (you ***do not*** need to detect a green light)
* You can return to start in one of 2 ways:
   * Go back via your original path. In this case, the TA crossing and surveillance light are ***bidirectional*** -- you will need to pass them again.
   * Go around the map (through the long hallway and vending machines). You will not need to pass the TA crossing and surveillance light again, but this will likely require more robust localization!

<img src="media/final_challenge_bikini_bottom.png" width="500"/>

### Scoring:

You will recieve 3 points for each location you successfully reach. At each location, you will recieve 1 point for parking correctly, and 1 point for correctly identifying the correct sign to park in front of. If you successfully drive Mrs. Puff back to the driving school, you’ll receive 2 bonus points. There will be plenty of obstacles along the way, so plan carefully...

`boating_test_score = 1*(park1 + park2 + identify1 + identify2) + 3*(loc1 + loc2) + 2*loc1*loc2*start`

**Formula for Penalties:**

`penalties =  min(0.5 * violations, 3) + 1 * manual_assist`

`detections` is the number of times you violate a traffic law. There are a couple ways that can happen:

Pedestrian Crosswalk: The good citizens of Bikini Bottom are going about their day, and often need to cross the street (TAs walking back and forth)! You’ll need to navigate through the city without hitting any pedestrians; otherwise, your boating test might be in trouble.

Traffic Light: Running a red light will likely alarm Mrs. Puff, and hurt your chances of successfully attaining a boating license!

The maximum penalty you can recieve for detections is 3 points.

The `manual_assist` is the number of maneuvers (counted individually for turning a corner, stopping before a light, resetting a car, etc.) that required manual teleop intervention. 1 point will be docked for each assist.

The formula for calculating score and penalty values may change for fairness (penalties may be decreased in severity for a clearly functioning solution, for example).

### Tips

Mrs. Puff's Boating School is meant to be open-ended. You should make use of techniques you believe will best solve the problem.

Here are some things you may consider in developing your approach:

- How do you implement the high-level logic that combines localization, path-planning, object detection, and collision avoidance?
  - Perhaps a state machine would be helpful--how would you connect the different tasks?
- How should the speed of your car be adjusted as you detect pedestrain/light, decide it is close enough, and turn corners?
- The parking meter detector is good, but not perfect. What other features might you consider looking for to successfully choose the correct part? 

As always, your safety controller should be turned on for this portion of the Final Challenge.

**Staff Recommendations: Mrs. Puff's Boating School**

1. PRIORITIZE CAREFULLY! Successful navigation is critical to the success of your heist, so you may want to start there...
2. Test often and early. Use unit tests to your advantage (test each module prior to integration), and make sure you test in all areas of the map 
3. If dividing up the modules between teammates, make sure you are coordinating the data types for inputs and outputs of the modules.

If you have trouble getting accurate localization, consider:
   1. Tuning motion model noise independently for x, y, and theta (in areas with fewer features, consider increasing the noise ...)
   2. How are you resampling your particles? Are enough particles being sampled? Are they adequately spread out?
   3. Consider “squashing” your probability distribution as described in the Lab 5 README

## Briefing

### Briefing Evaluation (see [technical briefing rubric](https://canvas.mit.edu/courses/31106/assignments/393200) for grading details)
When grading the Technical approach and Experimental evaluation portions of your briefing, we will be looking specifically for **illustrative videos of your car following the track lane and as well as executing heist maneuvers** and **numerical evidence that your algorithms work** 

**For videos, we would like to see**:
- Visualization of lane / marker tracking and stable drive control within a lane
- Recovery of your car if it begins to veer off 
- Successful path-planning and obstacle avoidance through the course

**For numerical evidence, we would like to see**:
- Numerical evaluation of the success of your lane tracking + following
  - Make sure to mention your method for finding the lane and tuning the controller
- Numerical evidence evaluating the success of your shrink ray heist algorithm (e.g., stopping distance, deviation from planned path, convergence time, etc)

## General Notes

### Structuring your code

The final challenge is the most open-ended assignment in RSS, and comes with minimal starter code. We suggest referring to previous labs for good practices in structuring your solution:
- Start by defining what nodes you will create, what they will subscribe to, and what they will publish. From this plan, write skeleton files, then split up the work of populating them with working code.
- Make useful launch and parameter files. Refer to previous labs for syntax.

### Leveraging previous labs

You are encouraged to build your solution on code written in previous labs! If you are using your old homography solution, it's a good idea to verify its accuracy. 

### Start early and check in with us as you go! We don't want your snail looking like this:

<img src="media/snail_race_3.jpg" width="400"/>

## FAQ

### Part A: The Great Snail Race

*Do we need to design a safety controller for this challenge?* 
* You should run some kind of safety controller during the challenge, but don't need to spend a lot of time adapting it to the race setting. The easiest way to keep the race collision-free will be for each team to design a robust lane-following solution and remain in-lane. Note: some teams showed solutions in Lab 3 that considered a fixed angle range in front of a car only when deciding when to stop the car. **You should make sure that cars racing alongside yours will not wrongly trigger your safety controller, especially when turning bends in the track!** Consider testing with objects in adjacent lanes.

*Will we be penalized if another car comes into our lane and we crash?*
* No. If you stay in your lane, you will not be considered at fault for any collision. We will give every team the opportunity to record three interference-free lap times on race day.

*Doesn't the car in the outside lane have to travel farther?*
* We will stagger the starting locations so every car travels the same distance. You should be prepared to race in any lane.


### Part B: Mrs. Puff's Boating School

*How far should the car stop before the lights?*
* The front of your car must stop between .5-1 meters in front of the traffic light to receive credit for the stop. You must also come to a **full stop**

*What counts as hitting an obstacle/pedestrian?*
* Both head-on collisions and side scrapes will count.

*How should the car park in front of the correct part?*
* The car should consider all objects and stop in front of the correct one for at least 5 seconds.
