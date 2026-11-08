# KSP Python Guidance Programs

Python code that flies rockets in Kerbal Space Program, from the Everyday Engineer video
**"Python Flies a Drop Pod, Cruise Missile and Ballistic Missile in KSP"**: [VIDEO LINK]

The three programs climb from simple to hard, and from no AI to almost all AI:

| File | Vehicle | Control | Size | Who wrote it |
|---|---|---|---:|---|
| `Drop Pod code.py` | Orbital drop pod | Open loop: deploys gear, fires the landing burn and throttles by altitude and speed | ~35 lines | Me, no AI |
| `ASM-03 PID V3.py` | Sea-skimming cruise missile | 4 PID loops (pitch, target pitch, roll, yaw), cruises 50 m above the water | ~300 lines | Mostly me, cleaned up with AI |
| `ballistic_missile.py` | Ballistic missile | Impact predictor: steers out the left/right error, cuts the engine to fix range, then trims with fins on reentry | ~4,200 lines | Written with Claude Code, under my direction |

## What you need

1. **Kerbal Space Program** (1.x)
2. **The kRPC mod**, which lets Python talk to KSP. The easiest way to install it is the CKAN mod manager.
3. **Python 3** and the kRPC client: `pip install krpc`

## How to run one

1. Start KSP, put your craft on the pad, and start the kRPC server (the kRPC window in game, then *Start server*).
2. **Cruise missile:** set an in-game target first (the code flies at `target_vessel`).
   **Ballistic missile:** set a target vessel or a waypoint named `Target`, or type coordinates into
   `TARGET_LAT` / `TARGET_LON` near the top of the file.
3. Run the file: `python "Drop Pod code.py"` (or one of the others).

The settings are at the top of each file with comments. The numbers are tuned for my craft,
so expect to retune them for yours. That's half the fun.

## Questions

Comment on the video and I'll answer.
