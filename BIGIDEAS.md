1. Using gravity as an input to physical reservoic computing

Could we have a robot that is either a soft body or a tensegirty bot.
Then we have it push forward then due to gravity and the ground it feels a load.
Basically respond to the ground and gravity then using points we combine with mechcanical computing to have a way to add up all of this.
For example I was initally thinking soft body but tensegriy then inside we have all the points connected through mehcanical computing to do linear readout.
Then this could be useful for a way to have a robot know how to move itself with no electronics.
If used on a mars robot it could move a way we tell it to and we coudl have it get power from heat like through the thermal night and day diff we could get mechanical power to move it.
Could we use genetic algorithm and train each one using reservoir computing to find a good structure.


2. Reservoir Computing using a motor

The test case I was thinking of was hovering a drone. Normally we would need lots of stuff to hover a drone.
But here the idea would be to replace that with the motor as the reservoir and how it responds to it's external stimuli.
We could have it hover and the motor would notice the difference when a gust of wind knocks it around.
Then using the high frequency ripples in the electricity we train to correct it.
Then all we need a simple linear readout and maybe a RLS to update live.
Instead of sampling once per control cycle we sampel thousands midway through PWM pulse
We can probably be faster than normal flight controller.
Got to check if the PWM electrical noise doesn't drown everything out
Compared to a simple NN.


We could do a python simulation of a nonlinear BLDC motor