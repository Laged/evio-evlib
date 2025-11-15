A. Challenge name and one-liner
● Challenge name/tagline (2-4 words max):
Lights, Camera, Reaction!
● Challenge one-liner (one sentence):
Operate in a timescale where ordinary vision fails - machine vision on
microsecond timescales.
B. Engage and excite participants with an
introduction
● Introduction to the challenge (3-4 sentences max):
There's a camera type you might have never heard about – an Event
Camera. They use an imaging sensor that detects changes in light
intensity at the pixel level, producing a stream of asynchronous
events rather than full frames. Their key advantages include very high
dynamic range, low latency (microseconds), minimal motion blur, and
high temporal resolution. Can you figure out how to use such a camera
to detect objects that move extremely fast? Prove it.
C. Describe the actual challenge
● The challenge (3-5 sentences max):
You’ll get access to a dataset of event camera recordings and your
mission is to build real-time ML models that can detect and interpret
motion from these microsecond-level vision streams.
Start simple: can you count how many times per second a fan spins?
Then push further - track drones, estimate speed, predict motion,
whatever you can imagine.
Push these sensors beyond their limits — and remember, their limits
are very high.
D. Give some initial analysis and insights on the
challenge
We’re looking for real-time detection of objects seen by the event
cameras. You will get a dataset of event camera footage and a simple
python package for event reading. The dataset includes footage of a
rotating fan and a flying drone. With the reading package named evio,
you can read the event files in a structured manner and generate a
paced iterator from the events. The repo includes an example script
for playing the event footage. If you want to later try accessing the
live cameras we brought, you can install the Prophesee Metavision SDK
(Available for Linux and Windows).
Start by familiarizing yourself with the event data by downloading the
dataset and using the scripts/play_dat.py to visualize the scene. Try
adjusting the window and speed arguments to see how this affects the
data playback. The play_dat.py is also a great resource to get started
with your own scripts.
E. Mentors and contact persons
● Who will be mentoring the challenge from your side?
Sensofusion engineers and researchers will be on location during
Junction to mentor and assist.
● Who will be the main contact person for the challenge?
Samuli Nyman, R&D Engineer, Sensofusion
F. Describe the technologies and support for the
challenges
● What You’ll Bring (Tech, software and/or mentors)
- Access to event camera recordings and event data I/O tool.
- Mentors from Sensofusion R&D team with expertise in vision
technology and AI.
- Example scripts to help teams get started.
G. The Prize and how to win it
● The Prize(s) (2-3 sentences max):
The winning team, as chosen by the Sensofusion mentors, will be
rewarded with DJI Mini 3 drones. Every member of the winning team will
get a drone.
● How you judge solutions (2-3 sentences max):
The most accurate, creative, technically impressive, and real-time
capable use of the event camera wins.
Go wild, push the limits of machine vision, and surprise us with what
you can make microseconds reveal.
H. Your Company and its areas of interest
● Information about the company (3-4 sentences max):
The mission for Sensofusion is to secure personnel and infrastructure
against drones with our Airfence product. Our systems are ready,
tested and shipping. Sensofusion has governmental customers in Europe,
North America, Middle East and Asia.
