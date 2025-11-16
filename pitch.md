# Submisson
## Project Name
weUseNixBtw

## Punchline
Real-time microsecond scale event processing for drone detection - it works on our machines (maybe yours)

## Description
###1) A unified GUI for exploring event-based camera data
###2) A preprocessing pipeline to transform proprietary binary data from .dat to industry-standard .h5
###3) A blazingly-fast event-data loader (yes, python bindings with rust backend)
###4) A robust, multi-layered computer vision algorithm for real-time RPM counting & drone dtection
###5) All packaged in an elegant flake.nix - just nix run .#evio-evlib

# Pitch

## Intro
I am Matti - these are Jesse & Irene - and together, weUseNixBtw. We hacked away at the "Light, camera, reaction!" challenge by Sensofusion.

I am not here to sell you drones. Or anti-drone tech. Ask Mikko about that. That's what he sold us, at least, a unique opportunity to improve national security. It should be easy they said, they have an amazing dataset prepared for us.
*open fan spinning data*


## The challenge
This is our biggest Fan. They asked as to tell them how fast it spins. Can't they see? Its RPM is over 9000! Well, they wanted us to be more specific, and perhaps get back to identifying drones as well at some point. So we did.
*slow down the time window param to show the fan rotos*

## The solution
We played around with the data and figured out how to properly batch event-based data into meaningful data. We applied *TODO IRENE'S MAGIC EXPLANATION*
*add fan mask overlay*

Then we did *TODO IRENE'S MAGIC EXPLANATION*
*add rotor bounding boxes and RPM calculation*

The result? THE RPM WAS OVER 9000! 11123 to be exact.

Well that was fun. How about if we vary the RPM?
*change data to var_rpm*
*TODO IRENE'S MAGIC EXPLANATION*

How about the drones?
*change data to hovering drone*
Okay! Our approach is the same, and the *TODO IRENE'S MAGIC EXPLANATION* does the trick.
*add masking / bbox / text overlays*

What if the drone moves? Doesn't matter.
*change data to moving drone*
Trees, birds, planes.. we don't care, we see the drone alone.
*play around with the params to show off*


## Solution approach
But remember we were not selling you drones nor anti-drone tech? We're here to sell you Nix! Our approach started with vibe coding the Friday away and scratching everything after three beers. On Saturday we invested HEAVILY on using nix to set up a shared development environemt that didn't only let us collaborate together, each one of us in our respective field. It also helped our agentic friends build *TODO CHECK OUT LoC FROM REPO* of first rate AI slop.

The key factor to make this all happen was a very strict nix environment where we were able to let the agents roam free.

## End punchline
Overall we started the hackathon without any prior computer vision experience, never having seen event based camera data, but very excited to spend the weekend at Junction. I'm pretty proud of what we built over the weekend - hopefully you learned something new - at least to try nix by running our repo locally with *TODO JESSE NIX PACKAGING MAGIC ie. nix run .#demo"

I'll be glad to answer any questions - or to forward you to Irene regarding drones/computer vision, or Jesse for nix wizarding.

Thank you, I'm Matti from weUseNixBtw!


