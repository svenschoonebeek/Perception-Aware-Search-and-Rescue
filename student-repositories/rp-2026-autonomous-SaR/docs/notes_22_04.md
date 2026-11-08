Desired metrics:
- Collisions with terrain (When agent tries to go out of bounds or hit a wall)
- Collisions with victims
- Collisions between agents
- Number of victims identified
- Time it took to find all victims
- Damage / amount of vulnerability


Agents will make their decisions based on current ground truth, then will act concurrently.

"move_history"

If you scan something, its certainty is incremented with a multiplier, so if you scan multiple times accurracy increases.

Stopping conditions:
- All victims are found
- Or a certain duration was reached.
- Certain damage acquired.

TODO:
- Vulnerability matrix (Tigo)
- Room in top left corner (Tigo)
- Continue visualizer and add play button (Maria)
- Implement scanning functionality (Filip & Nico)
- Doing metrics (Jacob)

Next meeting:
- Tuesday 28th of April after lecture (15:45)



