William Gilpin
Multiple Reservoir connections boosted gradient
Send data from each reservoir state to next then at end linear rgression with only last or all of them used.
Maybe predict difference instead of next actual value

What I Did:
Gonna use this now to know what I did lol and for the slides tbh
- Monday (06/29/26)
  - Stuff I did
    - Checked out different reservoirs from Sparse Random, Sparse Orthogonal, Delay Line in deep_reservoir.ipynb
  - Ideas
    - Lowkey the Delay Line and Orthogonal did pretty good and I was thinking a deep reservoir ends up basically doing both
    - Since each reservoir will have a form of Orthogonal but then at the end I'll use the state as inputs for next reservoir
    - I was thinking that we are basically doing a form of Delay Line just through each reservoir.
  - Things to do
    - Now I want to implement both a boosted reservoir and deep reservoir
    - Need to do Liquid State Machines