#!/usr/bin/env python3
r"""
=============================================================================
 KSP BALLISTIC MISSILE GUIDANCE  --  kRPC
=============================================================================

 Single-stage, fixed-thrust ballistic missile with aerodynamic terminal
 guidance.  Flies vertically off the pad, pitches over onto a loft angle,
 closed-loop steers in yaw to null crossrange, cuts the engine at the exact
 moment the predicted impact point crosses the target, coasts, then uses
 fins to trim out the residual error during reentry.

 HOW IT WORKS
 ------------
 Everything is built on one function: ImpactPredictor.predict().  It takes
 the missile's current position and velocity and numerically integrates them
 forward until they cross the target's altitude, returning where the missile
 would land if you stopped thrusting right now.  Guidance is then just:

     predict -> compare to target -> correct

 The two errors are handled by two completely separate controls:

     CROSSRANGE (left/right)  ->  steering (heading command)
     DOWNRANGE  (short/long)  ->  engine cutoff timing

 REFERENCE FRAME
 ---------------
 All propagation happens in body.reference_frame -- the frame that ROTATES
 with Kerbin.  The target is stationary in this frame, so no bookkeeping is
 needed for how far the planet turned during flight.  The price is that the
 equations of motion pick up two fictitious terms:

     a = -mu*r/|r|^3  -  2*(w x v)  -  w x (w x r)
         \___gravity_/    \Coriolis/    \_centrifugal_/

 KSP uses a LEFT-handed coordinate system, so the sign of w is not obvious.
 calibrate_omega() determines it empirically at startup by checking which
 sign reproduces the surface velocity kRPC reports.

 Body-frame axes (from the kRPC docs):
     x -> 0 deg latitude, 0 deg longitude
     y -> north pole
     z -> 0 deg latitude, 90 deg East

 USAGE
 -----
     1. Set your target in the SETTINGS block below (or set an in-game
        target / waypoint -- see TARGET_SOURCE).
     2. Put the missile on the pad, make sure kRPC is running.
     3. Run MODE = "attitude_test" on the pad.  If the vehicle cannot hold
        an attitude, nothing downstream matters.
     4. Run MODE = "predictor_test" while coasting on a hand-flown
        suborbital hop, and check the predicted impact point against where
        it actually lands.  Do NOT skip this -- every guidance stage is
        built on the predictor being right.
     5. Set MODE = "fly".

 WHERE THE REASONING LIVES
 -------------------------
 DESIGN_NOTES.md, next to this file.  The comments here say what a setting
 does and how to change it safely; the notes say why it is what it is, which
 flight measured it, and what was tried and abandoned.  Read the notes
 before moving anything.

 Requires: pip install krpc     (no numpy needed)
=============================================================================
"""

import csv
import datetime
import json
import math
import os
import sys
import time

VERSION = "2.2-dev"

try:
    import krpc
except ImportError:   # lets the offline test suite run without KSP
    krpc = None


# =============================================================================
# ||                              SETTINGS                                   ||
# =============================================================================
# Everything you should ever need to edit lives between here and END SETTINGS.
# Every non-obvious value was measured in flight; DESIGN_NOTES.md records which
# flight and why.  Changing one without reading the matching section is how
# regressions happen.

# --- Connection --------------------------------------------------------------
CONNECTION_NAME = "Ballistic Missile Guidance"
RPC_ADDRESS     = "127.0.0.1"   # kRPC server address (same PC = leave alone)
RPC_PORT        = 50000
STREAM_PORT     = 50001

# --- What to run -------------------------------------------------------------
# Exactly ONE of:
#   "fly"            - the full mission.
#   "attitude_test"  - hold one fixed attitude and report tracking.  Does NOT
#                      launch.  Start here whenever the vehicle misbehaves.
#   "predictor_test" - print one impact prediction and stop.  Only informative
#                      IN FLIGHT; on the pad the honest answer is "on the pad".
MODE = "fly"

CONFIRM_LAUNCH = True   # console prompt before ignition.  Only applies to
                        # "fly", and is ignored while the in-game panel is up
                        # -- there the CONFIRM LAUNCH button is the prompt.

# --- In-game control panel ---------------------------------------------------
# Draws a panel inside KSP (kRPC's UI service) to pick a target and launch,
# instead of the console Y/N prompt.  Only applies to MODE "fly".
# See DESIGN_NOTES "The in-game panel".
USE_GAME_PANEL = True
PANEL_UPDATE_HZ = 4.0    # panel refresh rate, INDEPENDENT of the guidance loop
                         # rate.  Every field is an RPC and the terminal phase
                         # runs at 20 Hz, so the panel must not keep up.
PANEL_LIST_VESSELS = True   # offer craft as targets, not just waypoints
PANEL_SURFACE_CRAFT_ONLY = False  # True = only landed/splashed/pad craft.
                         # False lists flying and orbiting craft too, marked
                         # MOVING -- their lat/lon is a snapshot and a shot
                         # takes ten minutes, so it will not be there.
PANEL_INCLUDE_DEBRIS = False      # debris is rarely what you meant to hit
PANEL_MAX_TARGETS = 60      # stop enumerating past this; a big save is slow

# --- Flight logging ----------------------------------------------------------
# Writes flights/<timestamp>_<mode>/ with transcript.txt, settings.json and
# summary.json, and appends one row to flights/results.csv.
# See DESIGN_NOTES "Flight logging".
FLIGHT_LOG = True
FLIGHT_LOG_DIR = "flights"      # relative to this file
FLIGHT_LOG_TEST_MODES = True    # also log attitude_test / predictor_test

# --- Target ------------------------------------------------------------------
# "manual"   : always use TARGET_LAT / TARGET_LON.
# "auto"     : in-game target vessel, else waypoint, else manual coords.
# "vessel"   : require an in-game target vessel.
# "waypoint" : require a waypoint named TARGET_WAYPOINT_NAME.
TARGET_SOURCE = "auto"

TARGET_LAT = -0.0972    # deg North (negative = South)
TARGET_LON = -74.5577   # deg East  (negative = West).  Default = KSC pad.
TARGET_WAYPOINT_NAME = "Target"

# --- Vehicle -----------------------------------------------------------------
# ENGINE_CAN_SHUTDOWN True  : range controlled by CUTOFF TIMING (accurate).
#                     False : true solid booster -- range controlled by LOFT
#                             ANGLE during boost instead.
ENGINE_CAN_SHUTDOWN = True

ENGINE_CAN_THROTTLE = True   # gates all three throttle mechanisms below
TERMINAL_THROTTLE   = 0.25   # throttle in the last seconds before cutoff, so
                             # each physics tick of timing error costs less
                             # range

# --- Late-boost throttle-down ------------------------------------------------
# Lowers guidance loop gain physically rather than by detuning.
# See DESIGN_NOTES "Late-boost throttle-down".
BOOST_THROTTLE_LOW = 1.00   # throttle for late boost.  1.0 disables.
BOOST_THROTTLE_AT  = 0.55   # trigger: predicted impact within this fraction
                            # of total range
BOOST_THROTTLE_MIN_MET = 30.0  # s: never throttle before this, so a wild early
                               # prediction cannot latch it during pitch-over
MAX_Q_MIN_THROTTLE = 0.40   # Q limiting may cut this deep and no deeper

# --- Ascent profile ----------------------------------------------------------
# See DESIGN_NOTES "Ascent shaping" for why 85 and not 90, and for the
# short-range scaling.
AUTO_SCALE_ASCENT = True

VERTICAL_PITCH = 85.0     # deg: NOT 90 -- at 90 the command carries no azimuth
VERTICAL_ALT   = 500.0    # m AGL: hold straight up until above this
VERTICAL_SPEED = 60.0     # m/s:   ...and moving at least this fast
PITCH_START_ALT = 800.0   # m: begin tilting over here
PITCH_END_ALT   = 18000.0 # m: fully on the loft angle by here

# Cap on how far the commanded attitude may sit from the velocity vector.
# Lower = safer but slower to fly the schedule.  Watch the "AoA LIMITED"
# markers.  See DESIGN_NOTES "Ascent AoA limits".
MAX_ASCENT_AOA = 15.0     # deg

# Above Q_SOFT_LIMIT the AoA limit tightens in proportion.  SIZE THIS NEAR THE
# VEHICLE'S ACTUAL MAX Q -- set too low and the limit sits on its floor for the
# whole ascent and the missile can never turn.
Q_SOFT_LIMIT   = 30000.0  # Pa
MIN_AOA_LIMIT  = 2.5      # deg: floor the tightening may reach

ASCENT_YAW_AOA_FLOOR = 6.0   # deg: yaw keeps its own larger floor, because
                             # crossrange is the error worth spending angle on

# Fin authority also fails at LOW Q.  Both collapse to the GIMBAL values when a
# gimbal is detected.  See DESIGN_NOTES "Control authority thresholds".
CONTROL_MIN_Q  = 2500.0   # Pa: below this fin authority is fading fast
FREEZE_STEER_Q = 600.0    # Pa: below this stop steering entirely and fly
                          # straight down the velocity vector
CONTROL_MIN_Q_GIMBAL  = 120.0
FREEZE_STEER_Q_GIMBAL = 40.0

# --- Pitch kick --------------------------------------------------------------
# A gravity turn cannot start itself.  See DESIGN_NOTES "Ascent shaping".
KICK_AOA       = 5.0      # deg below prograde to hold during the kick
KICK_END_PITCH = 82.0     # deg: kick until the VELOCITY vector reaches this
KICK_MAX_Q     = 32000.0  # Pa: bail out of the kick if it gets loud

# Loft angle (pitch above horizon) after pitch-over.
# None = minimum-energy:  45 - (range_angle / 4).
LOFT_ANGLE = None
LOFT_ANGLE_MIN = 20.0     # clamps for the auto value and for range control
LOFT_ANGLE_MAX = 85.0

# --- Range envelope ----------------------------------------------------------
# Printed in the launch check so an unreachable target shows up BEFORE
# ignition.  An estimate: +/-10%.  See DESIGN_NOTES "Range envelope".
ASCENT_LOSS_FRACTION = 0.29   # of ideal delta-v lost to gravity and drag
BURNOUT_SPEED_TO_PAD = 1.115  # the propagator launches from the pad, not from
                              # burnout at 52 km, so it needs more speed for
                              # the same distance
MIN_BURNOUT_SPEED = 970.0     # m/s at the closed-loop handover.
                              # VEHICLE-SPECIFIC: read it off the
                              # PITCH_PROGRAM -> CLOSED_LOOP_BOOST log line.
G0 = 9.80665                  # m/s^2, the Isp reference gravity

# --- Launch azimuth bias (Coriolis) ------------------------------------------
#   None : solve it before launch (recommended)
#   0.0  : disable, fly the raw great-circle bearing
#   else : force this bias in degrees, positive = steer RIGHT
#
# >>> TEMPORARILY FORCED TO 1.75.  Put this back to None once the solver
# >>> models the powered leg.  See DESIGN_NOTES "Coriolis and the launch
# >>> azimuth" -- 1.75 is a measurement for THIS vehicle, target and profile.
AZIMUTH_BIAS = None

AZIMUTH_SOLVE_ITERS = 6     # passes of the corrector.  Cheap: each is one
                            # drag-free speed bisection.
AZIMUTH_SOLVE_TOL = 200.0   # m: stop once the solved arc lands this close
AZIMUTH_BIAS_MAX = 15.0     # deg: refuse anything larger and fall back to no
                            # bias -- a huge one means the solver diverged

# --- Downrange aim offset ----------------------------------------------------
# Aim the ballistic solution long so terminal can dive short onto the target.
# Applied to the CUTOFF SOLUTION ONLY.  0 disables.
# See DESIGN_NOTES "The downrange aim offset".
AIM_OFFSET_DOWNRANGE = 7000.0   # m beyond the target

# --- Guidance gains ----------------------------------------------------------
# Boost crossrange steering, in DEGREES OF HEADING TRIM PER METRE of predicted
# crossrange miss.  Sized so 6000 m asks for roughly 12 deg of trim.
K_BOOST_XRANGE = 0.0020

# Range projection on the crossrange figure, so the same physical error reads
# the same early and late.  See DESIGN_NOTES "Crossrange: the range projection".
XRANGE_PROJECT = True      # False reverts to raw target-frame crossrange
XRANGE_PROJ_MAX = 6.0      # cap on the amplification (guards a 0/0 early on)
XRANGE_PROJ_MIN = 0.5      # floor, for when the predicted impact overshoots

TRIM_SMOOTHING = 0.15      # low-pass on the steering trim.  Lower = smoother
                           # and laggier.

# Lead on the boost crossrange loop:  crossrange + BOOST_LEAD_T * (its rate).
BOOST_LEAD_T      = 2.0    # s.  0 disables it.
BOOST_RATE_SMOOTH = 0.10   # smoothing on the measured crossrange rate
BOOST_RATE_MAX    = 400.0  # m/s: reject anything beyond this as noise
BOOST_LEAD_MAX_PT = 5.0    # deg of pointing error above which lead switches off

MAX_YAW_TRIM  = 25.0   # deg: cap on how far off the nominal bearing we steer
MAX_HEADING_RATE = 8.0 # deg/s: slew limit on the commanded heading
YAW_LOOP_MAX_PITCH = 85.0  # deg: do not steer in yaw while the velocity vector
                       # is steeper than this.  Too low and a LOFTED shot never
                       # unlocks crossrange steering at all.
XRANGE_MIN_FRACTION = 0.15 # ignore the crossrange solution until the predicted
                       # impact is this fraction of total range from the pad

K_LOFT        = 0.0004 # deg of loft change per metre of downrange error
                       # (only used when ENGINE_CAN_SHUTDOWN is False)

# --- Terminal guidance -------------------------------------------------------
# Both in DEGREES OF ANGLE PER METRE OF MISS.  They deliberately over-ask: the
# airframe is a spring at these pressures and a PD loop with no integral
# settles where the two balance.  See DESIGN_NOTES "Terminal guidance".
K_TERM_PITCH  = 0.0200 # deg of AoA per metre of downrange miss
K_TERM_YAW    = 0.0200 # deg of sideslip per metre of crossrange miss

# Gain scheduling on time-to-go:
#     scale = (TERMINAL_TGO_REF / tgo)^2, clamped to [1, TERMINAL_TGO_GAIN_MAX]
TERMINAL_TGO_REF      = 20.0  # s: time-to-go at which the raw gains apply
TERMINAL_TGO_GAIN_MAX = 3.0   # never multiply the gain by more than this

# Zero-effort-miss lead:  miss + LEAD * tgo * (closing rate).
# OFF BY DEFAULT -- needs a clean rate signal.  Turn on only once the reentry
# log shows a stable Cd*A and a smoothly varying miss.  Start around 0.3.
TERMINAL_LEAD        = 0.0
TERMINAL_LEAD_MAX_T  = 5.0    # s: cap on the lead time
TERMINAL_RATE_SMOOTH = 0.08   # smoothing on the measured closing rates
TERMINAL_RATE_MAX    = 150.0  # m/s: reject rates beyond this as noise

# Separate caps per axis.  Deflection comes out at ~0.10 per commanded degree,
# so 8 deg costs 0.80 of travel and leaves 0.20 for the PD loop.
# READ DESIGN_NOTES "Why the cap is 8 and not more" before raising these.
TERMINAL_AOA_MAX_PITCH  = 8.0   # deg
TERMINAL_AOA_MAX_YAW    = 12.0  # deg
TERMINAL_AOA_MAX_TOTAL  = 13.0  # deg: cap on the VECTOR SUM -- the real energy
                                # budget.  Crossrange is served first; note
                                # hypot(8,12) exceeds it, so the axes contend.

TERMINAL_AOA_MAX        = 12.0  # deg  (still used by the reentry trim)
TERMINAL_MAX_DEFLECTION = 1.00  # full control travel, 0..1

AOA_PEAK_TRACKING_TOL   = 1.5   # x commanded magnitude: how far past its
                                # command the vehicle may sit and still count
                                # as tracking.  Affects the printed advice
                                # only, never the flight.

TERMINAL_CMD_SMOOTHING = 0.25  # low-pass on the commanded angles, 0..1 per
                               # update; lower = smoother, laggier
TERMINAL_KD_SCALE      = 1.00  # multiplies ATT_KD during reentry/terminal.
                               # Raise if reentry alone rings.

AOA_MAX       = 8.0    # deg: hard cap on terminal angle of attack

# --- Phase transition thresholds ---------------------------------------------
# See DESIGN_NOTES "Phase transitions".
CLOSED_LOOP_MIN_FRACTION = 0.20  # start closed-loop boost once the predicted
                                 # range is this fraction of the target range
CUTOFF_WATCH_ERROR = 8000.0      # m: enter cutoff watch inside this error
TERMINAL_RANGE     = 15000.0     # m slant range: switch to terminal guidance
REENTRY_STEER_Q    = 20000.0     # Pa: hold prograde until the air is thick
                                 # enough to fly in

# --- Hands off in vacuum -----------------------------------------------------
# A statically stable airframe points itself into the wind for free; fighting
# for prograde in 2 Pa only builds momentum that overshoots when the air
# arrives.  See DESIGN_NOTES "Hands off in vacuum".
VACUUM_HANDS_OFF = True

# The bounded exception: a stable airframe will HOLD prograde once it is there
# but cannot ACQUIRE it in near-vacuum, so slew the nose forward late in the
# coast on a hard-capped deflection.
COAST_ALIGN = True
COAST_ALIGN_ABOVE = 20000.0    # m above the atmosphere interface at which to
                               # start aligning, while descending
COAST_ALIGN_DEFLECTION = 0.30  # fraction of ATT_MAX_DEFLECTION during the slew
                               # and through thin air afterwards.  Deliberately
                               # small: with no aerodynamic damping, the only
                               # protection against building momentum is not
                               # having the authority to build it with.
HANDS_OFF_UNTIL_Q = None       # Pa to resume attitude control.
                               # None = use the unpowered control_min_q.

# --- Predictor tuning --------------------------------------------------------
DT_VACUUM     = 2.0     # s: integration step above the atmosphere
DT_ATMOSPHERE = 0.25    # s: integration step inside the atmosphere
MAX_FLIGHT_TIME = 3000.0  # s: give up if it hasn't come down by now
DENSITY_SAMPLE_STEP = 500.0  # m: resolution of the atmosphere lookup table

INITIAL_CDA = 0.30   # m^2 Cd*A, used until the real one can be measured from
                     # flight.drag / flight.dynamic_pressure
CDA_SMOOTHING = 0.20 # how fast the measured drag area may move, 0..1 per update

# --- Which drag samples to believe -------------------------------------------
# See DESIGN_NOTES "Which samples to believe".  Watch 'raw' vs 'used' in the
# reentry log: if they never converge, the vehicle is not passing through
# alignment and none of this helps.
CDA_ALIGNED_ONLY   = True
CDA_MAX_SAMPLE_AOA = 12.0   # deg: how close to aligned counts as aligned
CDA_STALE_TIMEOUT  = 4.0    # s without an aligned sample -> accept anything
CDA_LOW_BIAS       = 0.00   # 0 = plain average, 1 = track the minimum.  0 on
                            # purpose: this airframe reenters nose-first, so
                            # the raw reading ALREADY is the aligned value.

# --- Two drag regimes --------------------------------------------------------
# One Cd*A cannot describe this descent -- it measured 0.24 high and 3.08 low
# at the same speed.  See DESIGN_NOTES "Drag: two regimes".
CDA_TWO_REGIME     = True
CDA_DENSE_ALT      = 25000.0  # m: below this the dense-air figure is used
CDA_DENSE_INITIAL  = 3.00     # m^2, prior; replaced once measured in flight
CDA_DENSE_MIN_Q    = 20000.0  # Pa: only sample the dense figure above this

# --- Loop rates (Hz) ---------------------------------------------------------
# The built-in attitude controller updates once per loop, so these also set how
# fast it runs.
RATE_OPEN_LOOP   = 20.0
RATE_CLOSED_LOOP = 10.0
RATE_CUTOFF      = 20.0
RATE_COAST       = 1.0
COAST_LOG_INTERVAL = 30.0   # s between coast log lines -- the coast can last
                            # half an hour
RATE_REENTRY     = 10.0
RATE_TERMINAL    = 20.0

# --- Attitude control --------------------------------------------------------
# False = use the built-in controller below (RECOMMENDED).
# True  = hand attitude to kRPC's autopilot, which controls all three axes and
#         has spun this vehicle to 90 deg/s.  See DESIGN_NOTES "Roll".
USE_KRPC_AUTOPILOT = False

# A PD loop on POINTING ERROR in degrees, producing deflection in -1..+1.
#   ATT_KP : push per degree of error.  Too low = sluggish, too high = ringing.
#   ATT_KD : damping per degree/second of BODY ROTATION.
# Tune Kp until it tracks, then Kd until the wobble stops.  No integral term.
# THE GAINS ARE PER-DEGREE: multiply by 180/pi before using the standard
# wn/zeta formulae.  See DESIGN_NOTES "Where ATT_KD comes from".
ATT_KP = 0.100
ATT_KD = 0.065     # zeta ~0.6 at the measured alpha of 0.53 rad/s^2

# What the D term measures.
#   True  : the rate the VEHICLE is rotating (command held fixed).  Sign is
#           correct by construction.
#   False : the numerical derivative of the error, which mixes vehicle rotation
#           with COMMAND motion.  Only safe at a small ATT_KD.
ATT_RATE_DAMPING = True
ATT_RATE_MAX = 200.0       # deg/s: reject anything past this as a glitch

ATT_KFF = 0.0              # command feed-forward, cancelling the lag body-rate
                           # damping adds.  ATT_KFF = ATT_KD reproduces the old
                           # controller exactly; try 0.3 * ATT_KD.

# --- Angle-of-attack feed-forward (descent only) -----------------------------
# A PD loop cannot HOLD a steady angle of attack -- with no integral it only
# produces steady output from a standing error.  This applies deflection
# proportional to the COMMAND instead.  Applied only where a trim reference
# exists: reentry and terminal.  See DESIGN_NOTES "AoA feed-forward".
ATT_AOA_FF = 0.10          # deflection per degree of commanded AoA.  0 = off.

# Deflection cap, scheduled on dynamic pressure at BOTH ends: it opens below
# ATT_DEFLECTION_Q and tapers as 1/Q above ATT_DEFLECTION_Q_HI, giving constant
# MOMENT rather than constant deflection.
# See DESIGN_NOTES "The deflection cap, both ends".
ATT_MAX_DEFLECTION = 0.5   # cap at high Q, 0..1
ATT_DEFLECTION_Q   = 3000.0     # Pa: below this the cap opens
ATT_DEFLECTION_Q_HI = 35000.0   # Pa: above this it tapers.  Just above the
                                # boost maximum, so ascent is unchanged.
ATT_DEFLECTION_MIN  = 0.35      # never taper below this much travel.  This is
                                # a SAFETY LIMIT, not a gain schedule -- the
                                # reentry floor, NOT the terminal cap, which is
                                # TERMINAL_MAX_DEFLECTION.

# --- Gain normalisation ------------------------------------------------------
# alpha (angular acceleration per unit input) is not constant, so scaling both
# gains by alpha_ref/alpha holds omega_n and zeta constant.  The reference is
# captured IN FLIGHT at the end of the vertical phase -- a gimbal produces no
# torque on the pad.  See DESIGN_NOTES "Gain normalisation".
GAIN_NORMALISE = True

# Which measurement to key off.
#   "inertia" : I_ref/I.      Clean and monotonic as the tanks drain.
#   "torque"  : alpha_ref/alpha.  Complete, but noisy under power.
#   "auto"    : inertia while burning, torque once out.  THE DEFAULT.
GAIN_BASIS = "auto"
GAIN_SCALE_MIN = 0.10      # never soften past this.  Boost never goes below
                           # 0.54; descent needs ~0.196 at alpha 2.7.
GAIN_SCALE_MAX = 2.50      # never stiffen past this

# Two time constants, because boost alpha tracks inertia draining over a
# hundred seconds while descent alpha tracks Q going ten-fold in eight.
ALPHA_TAU_DESCENT = 1.0    # s: unpowered
ALPHA_TAU      = 6.0       # s: powered

# Gimbal torque is proportional to thrust, so at 50% throttle the loop has half
# the authority it was tuned with.  OFF by default -- fin torque scales with Q
# instead, so this over-corrects whenever the fins contribute.  Turn on if
# pointing error roughly doubles and STAYS doubled after the THROTTLE DOWN line.
GAIN_THROTTLE_COMP = False

# Roll handling:
#   "free" : write a hard 0.0 to roll always.
#   "damp" : null the roll RATE, ignore the roll ANGLE.  Right for a missile --
#            steering is roll-agnostic, but a spinning vehicle cannot be
#            steered at all.  See DESIGN_NOTES "Roll".
ROLL_MODE = "damp"
ATT_KROLL = 0.015          # control input per degree/second of roll rate
ROLL_SIGN = 1.0            # starting guess; overwritten by the calibration

# --- Autopilot behaviour (only used when USE_KRPC_AUTOPILOT is True) ---------
# THE DIRECTIONS ARE NOT ALL THE SAME WAY ROUND -- see DESIGN_NOTES "The kRPC
# autopilot path".  The middle number of each PID triple is Ki; leave it zero.
AP_STOPPING_TIME     = (0.3, 0.3, 0.3)   # lower = gentler
AP_DECEL_TIME        = (8.0, 8.0, 8.0)   # higher = gentler
AP_ATTENUATION_ANGLE = (5.0, 5.0, 5.0)   # higher = gentler
AP_AUTO_TUNE         = False
AP_TIME_TO_PEAK      = (5.0, 5.0, 5.0)   # only used if AP_AUTO_TUNE is True
#                  (  Kp,  Ki,  Kd )
AP_PITCH_PID         = (1.2, 0.0, 2.0)
AP_YAW_PID           = (1.2, 0.0, 2.0)
AP_ROLL_PID          = (0.5, 0.0, 1.5)

# --- Control authority limits ------------------------------------------------
# Written straight to the craft, exactly like dragging the VAB sliders.
# None leaves the craft's own settings alone.  If the engine does not gimbal
# the fins are the ONLY control there is -- the startup banner reports how many
# gimbals it found.  See DESIGN_NOTES "The in-game authority limits".
CONTROL_SURFACE_AUTHORITY = 0.70   # fin deflection, 0..1
GIMBAL_LIMIT              = 1.00   # engine gimbal range, 0..1

DISABLE_ROLL_CONTROL = False   # False because the roll calibration pulse needs
                               # roll authority.  Safe: our controller writes
                               # control.roll explicitly every cycle.

# --- Roll alignment ----------------------------------------------------------
# Bank-to-turn alignment: OFF.  Steering does not need it and it cost ~100 deg/s
# of spin.  The short sign-calibration pulse still runs when ROLL_MODE is
# "damp", because the damper needs the sign and it cannot be derived.
ROLL_ALIGN           = False
# The aligner is a CASCADE, not a single PD -- roll angle is a double
# integrator and the rate cap is what keeps it well behaved.
ROLL_ALIGN_GAIN      = 0.80   # desired deg/s of roll per degree of error
ROLL_MAX_RATE        = 12.0   # deg/s: hard cap on commanded roll rate
ROLL_RATE_GAIN       = 0.020  # control input per deg/s of rate error
ROLL_ALIGN_MAX       = 0.25   # cap on roll deflection while aligning
ROLL_ALIGN_TOLERANCE = 10.0   # deg: close enough, stop fiddling
ROLL_ALIGN_TIMEOUT   = 15.0   # s: give up and carry on regardless
ROLL_CAL_INPUT       = 0.35   # size of the sign-calibration pulse
ROLL_CAL_TIME        = 1.0    # s: how long to hold the pulse
ROLL_CAL_MIN_RESPONSE = 1.0   # deg/s: below this the pulse told us nothing

# The first attempt fires at ~1.5 kPa, below CONTROL_MIN_Q, and has failed on
# every flight.  See DESIGN_NOTES "Calibration retry".
ROLL_CAL_RETRY       = True
ROLL_CAL_MIN_Q       = 10000.0  # Pa: don't bother retrying below this
ROLL_CAL_RETRY_INPUT = 0.15     # smaller pulse -- there is real air now
ROLL_CAL_MAX_TRIES   = 3        # give up for good after this many

# --- Max Q throttling --------------------------------------------------------
# Only usable when ENGINE_CAN_THROTTLE is True.  None disables it.
# PARKED at None to avoid flying a third simultaneous change; 25000.0 is the
# value to try next.  See DESIGN_NOTES "Parked and unfinished".
MAX_Q = None      # Pa

# --- Abort detection ---------------------------------------------------------
# A stable rocket weathercocks INTO the airflow, so AoA decays on its own.  If
# it instead grows and stays large the vehicle has departed and no amount of
# guidance will save it.  See DESIGN_NOTES "Aborts".
ABORT_ON_DEPARTURE = True
DEPARTURE_AOA      = 35.0   # deg
DEPARTURE_TIME     = 2.0    # s above DEPARTURE_AOA before calling it

ABORT_ON_DIVERGENCE = True
DIVERGENCE_STRIKES = 15   # consecutive failed/worsening predictions -> abort

# --- Attitude test settings (used when MODE = "attitude_test") ---------------
ATTITUDE_TEST_PITCH = 80.0
ATTITUDE_TEST_HEADING = 90.0
ATTITUDE_TEST_SECONDS = 30.0

# --- Misc --------------------------------------------------------------------
VERBOSE = True

# =============================================================================
# ||                            END SETTINGS                                 ||
# =============================================================================


# -----------------------------------------------------------------------------
# Flight logging.  See DESIGN_NOTES "Flight logging".
# -----------------------------------------------------------------------------

# Column order for results.csv.  Fixed: a reader that has already got rows will
# mis-align if this is reordered, so add to the END or start a new file.
_results_columns = [
    "started_utc", "version", "mode", "outcome", "note", "folder",
    "miss_m", "flight_time_s",
    "target_lat", "target_lon", "impact_lat", "impact_lon",
    "meco_met_s", "meco_downrange_m", "meco_crossrange_m",
    "meco_speed_ms", "meco_alt_m",
    "entry_spin_dps", "aoa_peak_deg",
    "cda_thin", "cda_dense",
    "range_km", "launch_bearing_deg", "azimuth_bias_deg", "aim_offset_m",
    "k_term_pitch", "k_term_yaw",
    "term_aoa_max_pitch", "term_aoa_max_yaw", "term_aoa_max_total",
    "att_kp", "att_kd", "att_aoa_ff", "terminal_kd_scale",
    "gain_basis", "max_q",
]


def _utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%SZ")


class _Tee(object):
    """Writes to the console and the transcript at once.

    Flushed on every write, so a crash, a Ctrl-C or a game revert still leaves
    a complete transcript up to that instant.
    """

    def __init__(self, console, handle):
        self._console = console
        self._handle = handle

    def write(self, text):
        self._console.write(text)
        try:
            self._handle.write(text)
            self._handle.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        self._console.flush()
        try:
            self._handle.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._console, "isatty", lambda: False)()


class FlightLog(object):
    """One folder per run, plus one appended row in results.csv.

    Call record() as numbers become known and finish() exactly once with the
    outcome.  finish() is idempotent, so it is safe in a `finally`.
    """

    def __init__(self, mode, base_dir=None):
        here = os.path.dirname(os.path.abspath(__file__))
        root = base_dir or os.path.join(here, FLIGHT_LOG_DIR)
        stamp = _utc_stamp()
        self.folder = os.path.join(root, "%s_%s" % (stamp, mode))
        # Two runs in the same second would otherwise collide.
        suffix = 1
        while os.path.exists(self.folder):
            suffix += 1
            self.folder = os.path.join(root, "%s_%s_%d" % (stamp, mode, suffix))
        os.makedirs(self.folder)
        self.results_path = os.path.join(root, "results.csv")

        self._handle = open(os.path.join(self.folder, "transcript.txt"),
                            "w", encoding="utf-8")
        self._console = sys.stdout
        sys.stdout = _Tee(self._console, self._handle)
        self._closed = False

        self.summary = dict.fromkeys(_results_columns)
        self.summary["started_utc"] = stamp
        self.summary["version"] = VERSION
        self.summary["mode"] = mode
        self.summary["folder"] = os.path.basename(self.folder)
        self.snapshot_settings()

    def snapshot_settings(self):
        """Dump every SETTINGS constant, read from the live module globals.

        Scanning globals rather than a hand-kept list means a setting added
        later is captured without anyone remembering to add it here.
        """
        snap = {}
        for name, value in sorted(globals().items()):
            if not name.isupper() or name.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            snap[name] = value
        path = os.path.join(self.folder, "settings.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, sort_keys=True)

    def record(self, **fields):
        """Merge known-so-far numbers into the summary.  Unknown keys are kept
        in summary.json but do not reach results.csv."""
        self.summary.update(fields)

    def finish(self, outcome, note=""):
        if self._closed:
            return
        self._closed = True
        self.summary["outcome"] = outcome
        if note:
            self.summary["note"] = note
        self.summary["ended_utc"] = _utc_stamp()
        try:
            print("\nflight log -> %s" % self.folder)
            with open(os.path.join(self.folder, "summary.json"), "w",
                      encoding="utf-8") as f:
                json.dump(self.summary, f, indent=2, sort_keys=True,
                          default=str)
            self._append_results()
        except Exception as exc:
            print("flight log: could not finish cleanly (%s)" % exc)
        finally:
            sys.stdout = self._console
            try:
                self._handle.close()
            except Exception:
                pass

    def _append_results(self):
        new_file = not os.path.exists(self.results_path)
        with open(self.results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(_results_columns)
            writer.writerow(["" if self.summary.get(c) is None
                             else self.summary.get(c)
                             for c in _results_columns])


# -----------------------------------------------------------------------------
# Vector helpers.  Everything is a plain 3-tuple of floats -- no numpy needed.
# -----------------------------------------------------------------------------

def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    """Standard right-handed cross product.

    KSP's frames are left-handed, so this does not have the geometric meaning
    you'd expect from a textbook.  That is fine: it is used ONLY inside the
    physics propagator, where the sign of omega is calibrated at runtime to
    make the whole thing self-consistent.  See calibrate_omega().
    """
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vmag(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vunit(a):
    m = vmag(a)
    return (a[0] / m, a[1] / m, a[2] / m) if m > 0 else (0.0, 0.0, 0.0)


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def pitch_heading_to_direction(pitch, heading):
    """(pitch, heading) in degrees -> unit vector in the SURFACE frame.

    vessel.surface_reference_frame axes are x = zenith (up), y = north,
    z = east, so this is straightforward spherical geometry.

    WHY THIS EXISTS instead of just calling target_pitch_and_heading():
    at 90 degrees pitch the direction is straight up no matter what heading
    you ask for -- heading is degenerate there.  kRPC's autopilot builds a
    full target ROTATION from pitch+heading, so it tries to satisfy the
    unreachable heading by rolling the vehicle, and pins the roll axis at
    full deflection indefinitely.  Commanding a bare direction instead leaves
    roll free, which is what you want on a rocket anyway.
    """
    p, h = math.radians(pitch), math.radians(heading)
    return (math.sin(p), math.cos(p) * math.cos(h), math.cos(p) * math.sin(h))


def wrap180(deg):
    """Fold an angle into [-180, +180)."""
    return (deg + 180.0) % 360.0 - 180.0


# -----------------------------------------------------------------------------
# Geodesy.  Converts between body-frame Cartesian vectors and lat/lon using
# the axis definitions from the kRPC CelestialBody docs.  Done locally rather
# than by calling body.latitude_at_position(), which would cost thousands of
# network round-trips per prediction.
# -----------------------------------------------------------------------------

def vec_to_latlon(v):
    """Body-frame vector -> (latitude, longitude) in degrees."""
    r = vmag(v)
    lat = math.degrees(math.asin(clamp(v[1] / r, -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[2], v[0]))
    return lat, lon


def latlon_to_vec(lat, lon, radius):
    """(latitude, longitude, radius) -> body-frame vector."""
    la, lo = math.radians(lat), math.radians(lon)
    return (radius * math.cos(la) * math.cos(lo),
            radius * math.sin(la),
            radius * math.cos(la) * math.sin(lo))


def local_axes(pos):
    """Up / north / east unit vectors in the body frame at a given position.

    Matches vessel.surface_reference_frame: x = zenith, y = north, z = east.
    """
    lat, lon = vec_to_latlon(pos)
    la, lo = math.radians(lat), math.radians(lon)
    up    = (math.cos(la) * math.cos(lo), math.sin(la), math.cos(la) * math.sin(lo))
    north = (-math.sin(la) * math.cos(lo), math.cos(la), -math.sin(la) * math.sin(lo))
    east  = (-math.sin(lo), 0.0, math.cos(lo))
    return up, north, east


def central_angle(a, b):
    """Angle at the planet's centre between two position vectors, in radians."""
    return math.acos(clamp(vdot(vunit(a), vunit(b)), -1.0, 1.0))


def bearing_to(pos, target, up=None, north=None, east=None):
    """Great-circle initial bearing from pos to target, degrees clockwise from
    north.  Same convention as the navball and autopilot heading."""
    if up is None:
        up, north, east = local_axes(pos)
    d = vsub(target, pos)
    horiz = vsub(d, vscale(up, vdot(d, up)))   # project onto local horizontal
    return math.degrees(math.atan2(vdot(horiz, east), vdot(horiz, north))) % 360.0


# -----------------------------------------------------------------------------
# Numerical integration
# -----------------------------------------------------------------------------

def blend_cda(current, raw, smoothing, low_bias):
    """Asymmetric smoothing that leans toward the LOW end of the signal.

    Falls fast, rises slowly.  On a signal that oscillates between an aligned
    (low drag) and a broadside (high drag) attitude, this settles near the
    bottom of the envelope instead of the middle -- which is what you want if
    the vehicle is passing through alignment and you intend to predict the
    aligned case.

    low_bias = 0 gives ordinary symmetric smoothing.
    """
    if raw < current:
        a = smoothing + low_bias * (1.0 - smoothing)
    else:
        a = max(0.01, smoothing * (1.0 - low_bias))
    return current + a * (raw - current)


def aim_offset(vhat, dp_deg, dy_deg):
    """A direction dp degrees above and dy degrees right of the velocity vector.

    vhat is a UNIT velocity in surface-frame components (x = up, y = north,
    z = east).  Returns a unit direction in the same frame.

    WHY THIS IS NOT JUST (pitch + dp, heading + dy)
    -----------------------------------------------
    Because heading stops meaning anything in a steep dive.  The horizontal
    part of a direction is cos(pitch), so at a -80 degree flight path angle
    it is 0.17: asking for 7 degrees of heading change moves the actual aim
    point by about one degree.  Terminal guidance would then quietly go slack
    exactly when it matters most, and the pointing error would look small
    while the missile sailed past the target sideways.

    Building the offset as perpendicular components added to the velocity
    vector gives the same angular authority at any dive angle, including
    straight down.
    """
    # "side" = 90 degrees right of the ground track, horizontal.
    hmag = math.sqrt(vhat[1] * vhat[1] + vhat[2] * vhat[2])
    if hmag < 1e-6:
        # Falling dead vertically: any horizontal direction is as good as
        # another, so pick north/east arbitrarily but consistently.
        side = (0.0, 0.0, 1.0)
    else:
        cn, ce = vhat[1] / hmag, vhat[2] / hmag
        side = (0.0, -ce, cn)

    # "lift" = local up with the along-velocity part removed, i.e. the
    # perpendicular direction the nose moves when you pitch up.
    up = (1.0, 0.0, 0.0)
    lift = vsub(up, vscale(vhat, vdot(up, vhat)))
    lm = vmag(lift)
    lift = vscale(lift, 1.0 / lm) if lm > 1e-6 else (0.0, 0.0, 0.0)

    tp = math.tan(math.radians(clamp(dp_deg, -80.0, 80.0)))
    ty = math.tan(math.radians(clamp(dy_deg, -80.0, 80.0)))
    return vunit(vadd(vhat, vadd(vscale(lift, tp), vscale(side, ty))))


def rk4_step(r, v, dt, accel_fn):
    """One classical 4th-order Runge-Kutta step for r'' = a(r, v).

    RK4 evaluates the acceleration four times per step and blends the results.
    Its error shrinks with the 4th power of the step size, so 2-second steps
    on a ballistic arc are accurate to well under a metre.
    """
    a1 = accel_fn(r, v)
    r2 = vadd(r, vscale(v,  dt * 0.5)); v2 = vadd(v, vscale(a1, dt * 0.5))
    a2 = accel_fn(r2, v2)
    r3 = vadd(r, vscale(v2, dt * 0.5)); v3 = vadd(v, vscale(a2, dt * 0.5))
    a3 = accel_fn(r3, v3)
    r4 = vadd(r, vscale(v3, dt));       v4 = vadd(v, vscale(a3, dt))
    a4 = accel_fn(r4, v4)

    dr = vscale(vadd(vadd(v, vscale(vadd(v2, v3), 2.0)), v4), dt / 6.0)
    dv = vscale(vadd(vadd(a1, vscale(vadd(a2, a3), 2.0)), a4), dt / 6.0)
    return vadd(r, dr), vadd(v, dv)


class AttitudeController:
    """A small PD attitude controller that writes the control axes directly.

    HOW IT WORKS
    ------------
    The target direction is expressed in the vessel's OWN reference frame,
    whose axes are (from the kRPC docs):

        x = out the right side
        y = forward, out the nose
        z = out the bottom

    So if the target direction lands at (tx, ty, tz) in those axes:

        the nose needs to come UP   by  atan2(-tz, ty)   (z points down)
        the nose needs to go RIGHT  by  atan2( tx, ty)

    Those two angles are the pitch and yaw errors, in the vessel's own frame,
    which is exactly the frame the control inputs act in.  No handedness
    guessing is required anywhere.

    WHERE THE DAMPING TERM COMES FROM
    ---------------------------------
    It damps BODY ROTATION, not the derivative of the error.  Those are only
    the same thing while the command is standing still, and this command
    slews at up to 8 deg/s.

    Reading the body rate the obvious way -- taking vessel.angular_velocity
    and resolving it into body axes -- means getting signs right in a
    left-handed coordinate system.  So it is obtained a different way, which
    needs no sign convention at all:

        keep last cycle's COMMAND direction, and ask what the error would be
        if the command had never moved

    Because the command is held fixed, everything that changed is the
    vehicle.  And because that hypothetical error is computed with exactly
    the same transform as the real one, it carries exactly the same sign
    convention -- whatever is correct for the P term is correct here too, by
    construction rather than by measurement.

    Cost: one extra transform_direction per cycle.  Worth it.  Set
    ATT_RATE_DAMPING = False to fall back to the old error derivative, which
    only works with a small Kd.

    WHY THERE IS NO INTEGRAL TERM
    -----------------------------
    An integrator accumulates whatever error it cannot remove and ramps its
    output until it saturates.  That is precisely the failure this controller
    was written to replace.  An attitude loop has no steady offset to trim
    out, so integral action buys nothing and costs a runaway.
    """

    def __init__(self, conn, vessel):
        self.sc = conn.space_center
        self.vessel = vessel
        self.e_pitch = 0.0
        self.e_yaw = 0.0
        # Rate at which body rotation alone is closing the pointing error,
        # deg/s per axis.  Negative while rotating towards the command.
        self.body_rate_pitch = 0.0
        self.body_rate_yaw = 0.0
        # Commanded angle of attack in body axes, for the feed-forward term.
        self.cmd_aoa_pitch = 0.0
        self.cmd_aoa_yaw = 0.0
        self._prev = None      # (ut, e_pitch, e_yaw, target_dir_surface)
        # When not None, written to the roll axis instead of the usual hard
        # zero.  Only the roll-alignment manoeuvre sets it.
        self.roll_override = None
        # Set by the calibration pulse -- measured, never assumed.
        self.roll_sign = ROLL_SIGN
        # Until the pulse has measured the sign, roll is left free rather
        # than damped: damping the wrong way is positive feedback.
        self.roll_sign_known = False
        # Roll rate in deg/s, fed in from streams by the guidance loop so the
        # controller does no RPC of its own.
        self.roll_rate = 0.0
        # Set from dynamic pressure by the guidance loop.  1.0 in thick air,
        # rising toward 2.0 as the air thins.
        self.deflection_boost = 1.0
        # Multiplies both Kp and Kd.  Set by the guidance loop from measured
        # torque and inertia; 1.0 means "as tuned".
        self.gain_scale = 1.0

    def update(self, target_dir_surface, ut, deflection_scale=1.0,
               kd_scale=1.0, trim_dir_surface=None):
        """Drive one control cycle.  target_dir_surface is a unit vector in
        the vessel's SURFACE reference frame (x=up, y=north, z=east)."""
        srf = self.vessel.surface_reference_frame
        own = self.vessel.reference_frame
        tx, ty, tz = self.sc.transform_direction(target_dir_surface, srf, own)
        # atan2 keeps this sane even when the target is behind us (ty < 0).
        self.e_pitch = math.degrees(math.atan2(-tz, ty))
        self.e_yaw = math.degrees(math.atan2(tx, ty))

        # -- damping term ----------------------------------------------------
        # d_pitch / d_yaw are the rate at which body rotation alone is closing
        # the pointing error, deg/s, negative while rotating towards the
        # command.  Same sign convention as the errors above, so "kp*e + kd*d"
        # damps rather than reinforces.
        d_pitch = d_yaw = 0.0        # body rotation
        f_pitch = f_yaw = 0.0        # command motion
        if self._prev is not None:
            dt = ut - self._prev[0]
            if 1e-3 < dt < 1.0:
                # Total rate of change of the error: body motion and command
                # motion mixed together.
                t_pitch = wrap180(self.e_pitch - self._prev[1]) / dt
                t_yaw = wrap180(self.e_yaw - self._prev[2]) / dt
                if ATT_RATE_DAMPING:
                    # Re-evaluate LAST cycle's command against the attitude we
                    # have NOW.  The command is held fixed, so the whole change
                    # is the vehicle rotating.
                    hx, hy, hz = self.sc.transform_direction(
                        self._prev[3], srf, own)
                    held_pitch = math.degrees(math.atan2(-hz, hy))
                    held_yaw = math.degrees(math.atan2(hx, hy))
                    d_pitch = wrap180(held_pitch - self._prev[1]) / dt
                    d_yaw = wrap180(held_yaw - self._prev[2]) / dt
                    # Whatever is left over is the command moving.
                    f_pitch = t_pitch - d_pitch
                    f_yaw = t_yaw - d_yaw
                else:
                    # One term, both effects, single gain.
                    d_pitch, d_yaw = t_pitch, t_yaw
                # A dropped frame or a frame flip can produce an enormous
                # apparent rate, so clamp rather than trust it.
                d_pitch = clamp(d_pitch, -ATT_RATE_MAX, ATT_RATE_MAX)
                d_yaw = clamp(d_yaw, -ATT_RATE_MAX, ATT_RATE_MAX)
                f_pitch = clamp(f_pitch, -ATT_RATE_MAX, ATT_RATE_MAX)
                f_yaw = clamp(f_yaw, -ATT_RATE_MAX, ATT_RATE_MAX)
        # -- angle-of-attack feed-forward ------------------------------------
        # trim_dir_surface is where the airframe would fly on its own
        # (prograde).  Resolving it through the SAME transform as the target
        # gives the commanded AoA directly in body axes, which matters because
        # bank is uncontrolled and a velocity-frame offset does not map onto
        # the body axes in any fixed way.  The difference is the angle the
        # airframe is being asked to hold.  See ATT_AOA_FF.
        a_pitch = a_yaw = 0.0
        if trim_dir_surface is not None and ATT_AOA_FF:
            gx, gy, gz = self.sc.transform_direction(trim_dir_surface, srf, own)
            a_pitch = wrap180(self.e_pitch - math.degrees(math.atan2(-gz, gy)))
            a_yaw = wrap180(self.e_yaw - math.degrees(math.atan2(gx, gy)))
        self.cmd_aoa_pitch = a_pitch
        self.cmd_aoa_yaw = a_yaw

        self.body_rate_pitch = d_pitch
        self.body_rate_yaw = d_yaw
        # Store the command too, so next cycle can hold it fixed.
        self._prev = (ut, self.e_pitch, self.e_yaw, target_dir_surface)

        lim = ATT_MAX_DEFLECTION * deflection_scale * self.deflection_boost
        ctl = self.vessel.control
        # Both gains carry the same scale, which keeps omega_n and zeta
        # unchanged as the vehicle lightens.
        kp = ATT_KP * self.gain_scale
        kd = ATT_KD * kd_scale * self.gain_scale
        kf = ATT_KFF * self.gain_scale
        ctl.pitch = clamp(kp * self.e_pitch + kd * d_pitch + kf * f_pitch
                          + ATT_AOA_FF * a_pitch, -lim, lim)
        ctl.yaw = clamp(kp * self.e_yaw + kd * d_yaw + kf * f_yaw
                        + ATT_AOA_FF * a_yaw, -lim, lim)

        # Roll.  In "free" mode this writes a hard zero every single cycle,
        # so nothing can wind up and nothing can spin the vehicle.
        if self.roll_override is not None:
            ctl.roll = clamp(self.roll_override, -1.0, 1.0)
        elif ROLL_MODE == "damp" and self.roll_sign_known:
            # Pure rate damping.  No angle term, so there is no roll attitude
            # to hold and nothing to wind up.
            ctl.roll = clamp(-self.roll_sign * ATT_KROLL * self.roll_rate,
                             -lim, lim)
        else:
            ctl.roll = 0.0

    def reset(self):
        """Forget the last cycle.  Call this after any gap in control.

        The damping term is a difference against the previous sample; across
        a gap that difference means nothing.  Also zeroes the reported error
        and rates so the log does not show a stale attitude the controller is
        no longer tracking.
        """
        self._prev = None
        self.e_pitch = self.e_yaw = 0.0
        self.body_rate_pitch = self.body_rate_yaw = 0.0
        self.cmd_aoa_pitch = self.cmd_aoa_yaw = 0.0

    @property
    def error(self):
        """Total pointing error in degrees, for logging."""
        return math.sqrt(self.e_pitch ** 2 + self.e_yaw ** 2)


class ImpactPredictor:
    """Propagates a state forward to impact in the rotating body frame."""

    def __init__(self, mu, radius, omega, atmo_top, density_table):
        self.mu = mu
        self.R = radius
        self.omega = omega
        self.atmo_top = atmo_top
        self.density_table = density_table   # (step, [rho0, rho1, ...])
        self.cda = INITIAL_CDA               # Cd * A, m^2 (measured in flight)
        # Dense-atmosphere Cd*A, used below CDA_DENSE_ALT.  See CDA_TWO_REGIME.
        self.cda_dense = CDA_DENSE_INITIAL
        self.mass = 1000.0                   # kg, refreshed each cycle
        self.use_drag = True

    # -- atmosphere ----------------------------------------------------------
    def density(self, alt):
        """Air density at an altitude, from the table sampled at startup."""
        step, table = self.density_table
        if alt <= 0.0:
            return table[0]
        i = int(alt / step)
        if i >= len(table) - 1:
            return 0.0
        frac = (alt - i * step) / step
        return table[i] + (table[i + 1] - table[i]) * frac

    # -- equations of motion -------------------------------------------------
    def accel(self, r, v):
        rm = vmag(r)
        # Gravity: points back at the planet's centre, falls off as 1/r^2.
        a = vscale(r, -self.mu / (rm * rm * rm))
        # Coriolis: deflects anything moving in a rotating frame.
        a = vsub(a, vscale(vcross(self.omega, v), 2.0))
        # Centrifugal: outward push from being in a rotating frame at all.
        a = vsub(a, vcross(self.omega, vcross(self.omega, r)))

        if self.use_drag:
            alt = rm - self.R
            if alt < self.atmo_top:
                rho = self.density(alt)
                sp = vmag(v)
                if rho > 1e-9 and sp > 1.0:
                    # Which drag regime are we propagating through?
                    # See CDA_TWO_REGIME.
                    cda = (self.cda_dense
                           if CDA_TWO_REGIME and alt < CDA_DENSE_ALT
                           else self.cda)
                    # a_drag = -0.5 * rho * |v|^2 * (Cd*A/m) * v_hat
                    #        = -0.5 * rho * |v|   * (Cd*A/m) * v
                    # In the rotating frame v is already air-relative, because
                    # the atmosphere rotates with the planet.
                    a = vadd(a, vscale(v, -0.5 * rho * sp * cda / self.mass))
        return a

    # -- the main event ------------------------------------------------------
    def predict(self, pos, vel, stop_alt):
        """Integrate forward until the trajectory descends through stop_alt.

        stop_alt is the TARGET's altitude above sea level, not the terrain
        under the flight path.  Only the terminal region matters, and this
        avoids an expensive terrain query on every single step.

        Returns (impact_position_vector, time_of_flight) or None if the
        missile never comes down inside MAX_FLIGHT_TIME.
        """
        r, v, t = pos, vel, 0.0
        stop_r = self.R + stop_alt

        while t < MAX_FLIGHT_TIME:
            alt = vmag(r) - self.R
            dt = DT_VACUUM if alt > self.atmo_top else DT_ATMOSPHERE

            pr, pv, pt = r, v, t
            r, v = rk4_step(r, v, dt, self.accel)
            t += dt

            if not math.isfinite(r[0]) or not math.isfinite(v[0]):
                return None   # integrator blew up

            # Impact = crossed below stop_r WHILE DESCENDING.  The descending
            # check stops it triggering on the launch pad, where we start out
            # below the target altitude but moving up.
            if vmag(r) < stop_r and vdot(r, v) < 0.0:
                return self._refine(pr, pv, pt, dt, stop_r)

        return None

    def _refine(self, r, v, t, dt, stop_r):
        """Re-walk the last step at 1/20th resolution to pin down the crossing."""
        fine = dt / 20.0
        for _ in range(21):
            nr, nv = rk4_step(r, v, fine, self.accel)
            if vmag(nr) < stop_r:
                # Linearly interpolate the last sliver for sub-step accuracy.
                h0, h1 = vmag(r) - stop_r, vmag(nr) - stop_r
                f = h0 / (h0 - h1) if h0 != h1 else 1.0
                return vadd(r, vscale(vsub(nr, r), f)), t + fine * f
            r, v, t = nr, nv, t + fine
        return r, t


# -----------------------------------------------------------------------------
# Startup calibration
# -----------------------------------------------------------------------------

def calibrate_omega(sc, body):
    """Work out the planet's angular velocity vector, sign included.

    KSP frames are left-handed, so we cannot just assume w = (0, +rate, 0).
    Instead: take a point sitting still on the equator in the rotating frame,
    ask kRPC what its velocity is in the non-rotating frame, and pick whichever
    sign of w reproduces that via v = w x r using OUR cross product.  Whatever
    handedness the game uses, the propagator is then internally consistent.
    """
    R = body.equatorial_radius
    p_rot = (R, 0.0, 0.0)                       # 0 N, 0 E, at sea level
    p_nr = sc.transform_position(p_rot, body.reference_frame,
                                 body.non_rotating_reference_frame)
    v_nr = sc.transform_velocity(p_rot, (0.0, 0.0, 0.0), body.reference_frame,
                                 body.non_rotating_reference_frame)
    rate = body.rotational_speed

    best_w, best_err = None, None
    for sign in (1.0, -1.0):
        w = (0.0, sign * rate, 0.0)
        err = vmag(vsub(vcross(w, p_nr), v_nr))
        if best_err is None or err < best_err:
            best_w, best_err = w, err
    return best_w, best_err


def build_density_table(body):
    """Sample the atmosphere once at startup so the propagator never has to
    touch the network mid-integration."""
    top = body.atmosphere_depth if body.has_atmosphere else 0.0
    n = int(top / DENSITY_SAMPLE_STEP) + 2
    table = [body.density_at(i * DENSITY_SAMPLE_STEP) for i in range(n)]
    return (DENSITY_SAMPLE_STEP, table), top


# Set by the in-game panel.  When present it wins over TARGET_SOURCE, so the
# panel does not have to fake up a TARGET_SOURCE mode to be obeyed.
_target_override = None


def use_target(target):
    """Aim at a target chosen from the panel.

    Also writes TARGET_LAT / TARGET_LON so the flight log's settings snapshot
    describes the shot that was actually flown, not the one in the file.
    """
    global _target_override, TARGET_LAT, TARGET_LON
    _target_override = target
    TARGET_LAT = target["lat"]
    TARGET_LON = target["lon"]


# KSP labels clustered waypoints with Greek letters rather than numbers.
# Leading underscore keeps it out of the flight log's settings snapshot.
_GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
         "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
         "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"]


def list_targets(sc, body):
    """Everywhere on `body` this missile could be aimed, for the panel to cycle.

    Three sources, in the order they are offered:
      * the manual coordinates from SETTINGS, always first so there is
        something to fly at even in an empty save;
      * every waypoint on this body -- contract markers and ones placed by
        hand both appear here;
      * every other vessel sitting on the surface (PANEL_LIST_VESSELS).

    Each entry is {"name", "lat", "lon", "source"}.  A source that raises is
    skipped with a note rather than taking the whole list down with it.
    """
    targets = [{"name": "Manual coordinates",
                "lat": TARGET_LAT, "lon": TARGET_LON, "source": "SETTINGS"}]

    try:
        for wp in sc.waypoint_manager.waypoints:
            if len(targets) >= PANEL_MAX_TARGETS:
                break
            if wp.body.name != body.name:
                continue
            name = wp.name
            if wp.clustered:
                name = "%s %s" % (name, _GREEK[wp.index % len(_GREEK)])
            targets.append({"name": name, "lat": wp.latitude,
                            "lon": wp.longitude, "source": "waypoint"})
    except Exception as exc:
        print("target list: waypoints unavailable (%s)" % exc)

    if PANEL_LIST_VESSELS:
        targets.extend(_vessel_targets(sc, body,
                                       PANEL_MAX_TARGETS - len(targets)))

    return targets


# Situations where a craft is sitting still, so a lat/lon taken now is still
# true when the missile arrives ten minutes later.
_PARKED = ("landed", "splashed", "pre_launch", "prelaunch", "docked")


def _enum_name(value):
    """Name of a kRPC enum member, without depending on how it is spelled.

    kRPC generates its enums at runtime and the spelling has moved between
    versions, so comparing names is safer than looking attributes up on the
    service object.
    """
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()


def _vessel_targets(sc, body, room):
    """Every craft on `body`, parked ones first.

    Each vessel is read inside its OWN try, because one odd craft must not cost
    the whole list -- that is exactly how `Vessel.body` (which does not exist;
    it is `orbit.body`) silently removed every craft from the panel.
    """
    parked, moving = [], []
    try:
        active = sc.active_vessel
        vessels = sc.vessels
    except Exception as exc:
        print("target list: vessels unavailable (%s)" % exc)
        return []

    for v in vessels:
        if len(parked) + len(moving) >= room:
            break
        try:
            if v == active:
                continue
            if v.orbit.body.name != body.name:
                continue
            kind = _enum_name(v.type)
            if kind == "debris" and not PANEL_INCLUDE_DEBRIS:
                continue
            situation = _enum_name(v.situation)
            at_rest = situation in _PARKED
            if not at_rest and PANEL_SURFACE_CRAFT_ONLY:
                continue
            f = v.flight(body.reference_frame)
            entry = {"name": v.name, "lat": f.latitude, "lon": f.longitude,
                     "source": "%s - %s" % (kind, situation),
                     "moving": not at_rest}
            (parked if at_rest else moving).append(entry)
        except Exception as exc:
            print("target list: skipped a craft (%s)" % exc)

    return parked + moving


def resolve_target(sc, body):
    """Get the target lat/lon according to TARGET_SOURCE.  Returns (lat, lon)."""
    if _target_override is not None:
        t = _target_override
        return t["lat"], t["lon"], "%s (in-game panel)" % t["name"]

    src = TARGET_SOURCE.lower()

    if src in ("vessel", "auto"):
        tv = sc.target_vessel
        if tv is not None:
            f = tv.flight(body.reference_frame)
            return f.latitude, f.longitude, "%s (in-game target vessel)" % tv.name
        if src == "vessel":
            raise RuntimeError("TARGET_SOURCE='vessel' but no target vessel set.")

    if src in ("waypoint", "auto"):
        for wp in sc.waypoint_manager.waypoints:
            if wp.name == TARGET_WAYPOINT_NAME and wp.body.name == body.name:
                return (wp.latitude, wp.longitude,
                        "%s (waypoint)" % wp.name)
        if src == "waypoint":
            raise RuntimeError("No waypoint named '%s' on %s."
                               % (TARGET_WAYPOINT_NAME, body.name))

    return TARGET_LAT, TARGET_LON, "manual coordinates from SETTINGS"


class MissionPanel(object):
    """A control panel drawn inside KSP, using kRPC's UI service.

    Cycle targets and launch without leaving the game window, then watch the
    flight and abort it from the same panel.

    COST.  Every text write and every button read is an RPC round-trip, and
    the terminal phase runs the guidance loop at 20 Hz.  So the panel refreshes
    at PANEL_UPDATE_HZ regardless of loop rate, never rewrites a field whose
    content has not changed, and reads its buttons from streams (which the
    server pushes) rather than polling them.

    FAILURE.  kRPC's UI service has broken across KSP releases before.  If
    anything here raises, the panel removes itself and the flight carries on in
    the console -- a broken panel must never cost a launch.
    """

    W, H = 280, 330
    GREEN = (0.45, 1.00, 0.45)
    RED   = (1.00, 0.40, 0.40)
    AMBER = (1.00, 0.80, 0.25)
    WHITE = (1.00, 1.00, 1.00)
    GREY  = (0.70, 0.70, 0.70)

    def __init__(self, conn):
        self.conn = conn
        self.panel = None
        self.ok = False
        self._texts = []
        self._streams = []
        self._cache = {}
        self._labels = None
        self._last_update = 0.0
        try:
            self._build()
            self.ok = True
        except Exception as exc:
            print("in-game panel: could not be created (%s)" % exc)
            print("   Carrying on in the console.  Set USE_GAME_PANEL = False")
            print("   to stop trying; kRPC's UI API does break between KSP")
            print("   versions and this is what that failure looks like.")
            self.remove()

    # -- construction ---------------------------------------------------------
    def _build(self):
        canvas = self.conn.ui.stock_canvas
        screen_w, _ = canvas.rect_transform.size
        self.panel = canvas.add_panel()
        self.panel.rect_transform.size = (self.W, self.H)
        # Hard against the right edge, vertically centred: clear of the stock
        # altimeter at the top and the staging stack on the left.
        self.panel.rect_transform.position = (screen_w / 2.0 - self.W / 2.0 - 20.0,
                                              0.0)

        y = self.H / 2.0 - 26.0
        for i in range(6):
            t = self.panel.add_text("")
            t.rect_transform.size = (self.W - 28, 22)
            t.rect_transform.position = (0.0, y)
            t.size = 16 if i == 0 else 14
            t.color = self.WHITE
            try:
                t.alignment = self.conn.ui.TextAnchor.middle_left
            except Exception:
                pass       # centred is ugly but harmless
            self._texts.append(t)
            y -= 23.0

        self.b_prev, self.s_prev = self._button("< PREV", -68.0, -34.0, 124, 30)
        self.b_next, self.s_next = self._button("NEXT >", 68.0, -34.0, 124, 30)
        self.b_go, self.s_go = self._button("LAUNCH", 0.0, -78.0, 256, 38)
        self.b_stop, self.s_stop = self._button("CANCEL", 0.0, -120.0, 256, 30)

    def _button(self, label, x, y, w, h):
        b = self.panel.add_button(label)
        b.rect_transform.size = (w, h)
        b.rect_transform.position = (x, y)
        stream = self.conn.add_stream(getattr, b, "clicked")
        self._streams.append(stream)
        return b, stream

    # -- primitives -----------------------------------------------------------
    def _take(self, stream, button):
        """True once per press.

        kRPC LATCHES `clicked` until it is written back to False, so a press
        that is not consumed here would read as a press again next cycle.
        """
        try:
            if stream():
                button.clicked = False
                return True
        except Exception:
            pass
        return False

    def _write(self, row, content, color=None):
        """Set one line, skipping the RPC when nothing changed."""
        key = (content, color)
        if self._cache.get(row) == key:
            return
        self._cache[row] = key
        try:
            self._texts[row].content = content
            if color is not None:
                self._texts[row].color = color
        except Exception:
            pass

    def _label(self, prev, nxt, go, stop):
        if self._labels == (prev, nxt, go, stop):
            return
        self._labels = (prev, nxt, go, stop)
        try:
            self.b_prev.text.content = prev
            self.b_next.text.content = nxt
            self.b_go.text.content = go
            self.b_stop.text.content = stop
        except Exception:
            pass

    def _drain(self):
        """Discard any presses that landed while we were not watching."""
        for stream, button in ((self.s_prev, self.b_prev),
                               (self.s_next, self.b_next),
                               (self.s_go, self.b_go),
                               (self.s_stop, self.b_stop)):
            self._take(stream, button)

    # -- pre-launch -----------------------------------------------------------
    def choose_target(self, targets, launch_pos, radius, start=0):
        """Cycle targets until SELECT or CANCEL.  Returns an index, or None.

        Blocking, and that is fine: nothing has been ignited yet, so there is
        no loop deadline to miss.  10 Hz feels responsive and costs nothing.
        """
        index = start % len(targets)
        self._write(0, "GBM-01   %s" % VERSION, self.AMBER)
        self._label("< PREV", "NEXT >", "SELECT", "CANCEL")
        self._drain()
        while True:
            t = targets[index]
            tgt = latlon_to_vec(t["lat"], t["lon"], radius)
            rng_km = radius * central_angle(launch_pos, tgt) / 1000.0
            brg = bearing_to(launch_pos, tgt)
            self._write(1, "TARGET  %d of %d" % (index + 1, len(targets)))
            self._write(2, t["name"][:28], self.GREEN)
            self._write(3, "%+8.4f N %+9.4f E" % (t["lat"], t["lon"]))
            self._write(4, "%.1f km   bearing %.1f" % (rng_km, brg))
            if t.get("moving"):
                # Its lat/lon is a snapshot and the flight takes ten minutes.
                self._write(5, "%s - MOVING" % t["source"], self.RED)
            else:
                self._write(5, t["source"], self.GREY)

            if self._take(self.s_prev, self.b_prev):
                index = (index - 1) % len(targets)
            if self._take(self.s_next, self.b_next):
                index = (index + 1) % len(targets)
            if self._take(self.s_go, self.b_go):
                return index
            if self._take(self.s_stop, self.b_stop):
                return None
            time.sleep(0.1)

    def arm(self, g):
        """Show the launch check and wait.

        Returns "launch", "back" (pick a different target) or "abort".
        """
        rng_km = g.range_total / 1000.0
        self._write(0, "GBM-01   ARMED", self.AMBER)
        self._write(1, "TARGET")
        self._write(2, g.tgt_name[:28], self.GREEN)
        self._write(3, "%.1f km  brg %.1f  bias %+.2f"
                    % (rng_km, g.launch_bearing, g.azimuth_bias))
        env = None
        try:
            env = g.range_envelope()
        except Exception:
            pass
        if env:
            lo_km, hi_km = env[0] / 1000.0, env[1] / 1000.0
            inside = lo_km <= rng_km <= hi_km
            self._write(4, "envelope %.0f - %.0f km" % (lo_km, hi_km))
            self._write(5, "IN RANGE" if inside else "OUT OF RANGE -- WILL MISS",
                        self.GREEN if inside else self.RED)
        else:
            self._write(4, "envelope unavailable")
            self._write(5, "", self.GREY)
        self._label("BACK", "BACK", "CONFIRM LAUNCH", "CANCEL")
        self._drain()
        while True:
            if self._take(self.s_go, self.b_go):
                return "launch"
            if self._take(self.s_stop, self.b_stop):
                return "abort"
            if (self._take(self.s_prev, self.b_prev)
                    or self._take(self.s_next, self.b_next)):
                return "back"
            time.sleep(0.1)

    # -- in flight ------------------------------------------------------------
    def start_flight(self):
        self._write(0, "GBM-01   FLYING", self.AMBER)
        self._label("--", "--", "--", "ABORT")
        self._drain()

    def update(self, g):
        """One rate-limited refresh.  Safe to call every guidance cycle."""
        now = time.time()
        if now - self._last_update < 1.0 / PANEL_UPDATE_HZ:
            return
        self._last_update = now
        try:
            self._write(1, "%-14s %7.1f s" % (g.state, g.met()))
            self._write(2, "alt %6.0f m  v %5.0f m/s"
                        % (g.s_alt(), g.s_spd()), self.WHITE)
            self._write(3, "Q %6.0f Pa  AoA %+5.1f" % (g.s_q(), g.s_aoa()))
            sol = g._last_sol
            if sol is None:
                self._write(4, "no solution", self.GREY)
                self._write(5, "")
            else:
                self._write(4, "miss %8.0f m" % sol["miss"])
                self._write(5, "dn %+7.0f  cr %+7.0f"
                            % (sol["downrange"], sol["crossrange"]), self.GREY)
        except Exception:
            pass          # a panel that cannot draw must not stop the flight

    def abort_requested(self):
        return self._take(self.s_stop, self.b_stop)

    def finished(self, outcome, miss=None):
        self._write(0, "GBM-01   %s" % outcome.upper(),
                    self.GREEN if outcome == "impact" else self.RED)
        if miss is not None:
            self._write(1, "miss %.1f m" % miss)
        self._label("--", "--", "--", "CLOSE")

    # -- teardown -------------------------------------------------------------
    def remove(self):
        for stream in self._streams:
            try:
                stream.remove()
            except Exception:
                pass
        self._streams = []
        if self.panel is not None:
            try:
                self.panel.remove()
            except Exception:
                pass
            self.panel = None
        self.ok = False


# -----------------------------------------------------------------------------
# The guidance computer
# -----------------------------------------------------------------------------

class Guidance:

    # State machine states, in flight order.
    PRELAUNCH   = "PRELAUNCH"
    VERTICAL    = "VERTICAL_BOOST"
    ROLLALIGN   = "ROLL_ALIGN"
    KICK        = "PITCH_KICK"
    PITCHOVER   = "PITCH_PROGRAM"
    CLOSED_LOOP = "CLOSED_LOOP_BOOST"
    CUTOFF      = "CUTOFF_WATCH"
    COAST       = "COAST"
    REENTRY     = "REENTRY"
    TERMINAL    = "TERMINAL"
    IMPACT      = "IMPACT"
    ABORT       = "ABORT"

    def __init__(self, conn):
        self.conn = conn
        self.sc = conn.space_center
        self.vessel = self.sc.active_vessel
        self.body = self.vessel.orbit.body
        self.ref = self.body.reference_frame
        self.R = self.body.equatorial_radius

        print("ballistic_missile v%s" % VERSION)
        print("Initialising guidance on '%s' at %s"
              % (self.vessel.name, self.body.name))

        # --- target ---------------------------------------------------------
        self.tgt_lat, self.tgt_lon, self.tgt_name = resolve_target(
            self.sc, self.body)
        print("  target: %s" % self.tgt_name)
        self.tgt_alt = self.body.surface_height(self.tgt_lat, self.tgt_lon)
        self.tgt_pos = latlon_to_vec(self.tgt_lat, self.tgt_lon,
                                     self.R + self.tgt_alt)
        print("  lat %.5f  lon %.5f  elevation %.0f m"
              % (self.tgt_lat, self.tgt_lon, self.tgt_alt))

        # --- physics setup ---------------------------------------------------
        omega, omega_err = calibrate_omega(self.sc, self.body)
        print("  omega = (%.3e, %.3e, %.3e) rad/s   residual %.4f m/s"
              % (omega[0], omega[1], omega[2], omega_err))
        if omega_err > 1.0:
            print("  !! WARNING: omega calibration residual is large.")
            print("     Coriolis terms may be wrong -- expect crossrange bias.")

        print("  sampling atmosphere...")
        density_table, atmo_top = build_density_table(self.body)
        self.atmo_top = atmo_top

        self.pred = ImpactPredictor(self.body.gravitational_parameter,
                                    self.R, omega, atmo_top, density_table)

        # --- sanity check: our local lat/lon math vs the game's --------------
        p = self.vessel.position(self.ref)
        our_lat, our_lon = vec_to_latlon(p)
        ksp_lat = self.body.latitude_at_position(p, self.ref)
        ksp_lon = self.body.longitude_at_position(p, self.ref)
        dl = abs(our_lat - ksp_lat) + abs(wrap180(our_lon - ksp_lon))
        print("  lat/lon self-check error: %.6f deg %s"
              % (dl, "OK" if dl < 0.01 else "<-- PROBLEM"))

        # --- telemetry streams -----------------------------------------------
        # Streams push values continuously instead of being polled.  Without
        # them RPC latency alone would cap the loop rate.
        fl = self.vessel.flight(self.ref)
        self.s_pos  = conn.add_stream(self.vessel.position, self.ref)
        self.s_vel  = conn.add_stream(self.vessel.velocity, self.ref)
        self.s_mass = conn.add_stream(getattr, self.vessel, "mass")
        self.s_alt  = conn.add_stream(getattr, fl, "mean_altitude")
        self.s_srf  = conn.add_stream(getattr, fl, "surface_altitude")
        self.s_q    = conn.add_stream(getattr, fl, "dynamic_pressure")
        self.s_drag = conn.add_stream(getattr, fl, "drag")
        self.s_spd  = conn.add_stream(getattr, fl, "speed")
        self.s_ut   = conn.add_stream(getattr, self.sc, "ut")

        # Control-input streams: what the autopilot is actually commanding.
        # Small inputs on a wrong heading = guidance is wrong.  Inputs pinned
        # at +/-1.0 = the airframe is unflyable.
        ctl = self.vessel.control
        self.s_cpitch = conn.add_stream(getattr, ctl, "pitch")
        self.s_cyaw   = conn.add_stream(getattr, ctl, "yaw")
        self.s_croll  = conn.add_stream(getattr, ctl, "roll")
        self.s_aoa    = conn.add_stream(getattr, fl, "angle_of_attack")
        self.s_slip   = conn.add_stream(getattr, fl, "sideslip_angle")
        self.s_omega  = conn.add_stream(self.vessel.angular_velocity, self.ref)
        # Angular velocity and nose direction, both in the SURFACE frame.
        # Their dot product is the roll rate, with no per-cycle RPC.
        srf = self.vessel.surface_reference_frame
        self.s_omega_srf = conn.add_stream(self.vessel.angular_velocity, srf)
        self.s_nose_srf = conn.add_stream(self.vessel.direction, srf)
        self.s_moi = conn.add_stream(getattr, self.vessel, "moment_of_inertia")
        self.s_torque = conn.add_stream(getattr, self.vessel,
                                        "available_torque")

        self.att = AttitudeController(conn, self.vessel)
        self._want_dir = (1.0, 0.0, 0.0)   # last commanded direction

        self.ap = self.vessel.auto_pilot
        self.ap.reference_frame = self.vessel.surface_reference_frame
        self.ap.stopping_time = AP_STOPPING_TIME
        self.ap.deceleration_time = AP_DECEL_TIME
        self.ap.attenuation_angle = AP_ATTENUATION_ANGLE
        self.ap.auto_tune = AP_AUTO_TUNE
        if AP_AUTO_TUNE:
            self.ap.time_to_peak = AP_TIME_TO_PEAK
        else:
            self.ap.pitch_pid_gains = AP_PITCH_PID
            self.ap.yaw_pid_gains = AP_YAW_PID
            self.ap.roll_pid_gains = AP_ROLL_PID

        self.limit_control_authority()

    def limit_control_authority(self):
        """Turn down fin deflection and engine gimbal range in-game.

        This is the blunt, effective answer to "the controls are too
        aggressive": rather than retuning a controller, give the controller
        less authority.  Same thing as dragging the authority slider down in
        the VAB, but done from here so it is recorded with the rest of the
        settings.

        Wrapped in try/except throughout -- these are optional tuning knobs
        and a missing part type or renamed property must never take down a
        flight.
        """
        # KSP's own field is a 0-100 percentage, so that is what is written.
        # The value is read back and printed so wrong units are visible.
        n_surf, n_roll, back, surf_notes = 0, 0, [], []
        try:
            for cs in self.vessel.parts.control_surfaces:
                if CONTROL_SURFACE_AUTHORITY is not None:
                    try:
                        cs.authority_limiter = CONTROL_SURFACE_AUTHORITY * 100.0
                        back.append(cs.authority_limiter)
                        n_surf += 1
                    except Exception as exc:
                        surf_notes.append("!! authority: %s" % exc)
                if DISABLE_ROLL_CONTROL:
                    try:
                        cs.roll_enabled = False
                        n_roll += 1
                    except Exception as exc:
                        surf_notes.append("!! roll_enabled: %s" % exc)
        except Exception as exc:
            surf_notes.append("!! could not read control surfaces: %s" % exc)
        self.n_surfaces = n_surf
        if CONTROL_SURFACE_AUTHORITY is not None:
            print("  control surfaces: %d limited, readback %s"
                  % (n_surf, ["%.1f" % v for v in back[:4]] or "none"))
        if DISABLE_ROLL_CONTROL:
            print("  roll control disabled on %d control surfaces" % n_roll)
        for note in surf_notes:
            print("    %s" % note)

        # NOTE ON THE API: there is no Gimbal object.  The controls live
        # directly on Engine as gimballed / gimbal_locked / gimbal_limit /
        # gimbal_range.  Failures are reported noisily rather than swallowed,
        # because a missed gimbal silently disables every gimbal-aware setting
        # for the whole flight.
        n_gim, gim_notes = 0, []
        try:
            for e in self.vessel.parts.engines:
                try:
                    if not e.gimballed:
                        continue
                    n_gim += 1
                    # A locked gimbal reports a limit of 0 and does nothing.
                    if e.gimbal_locked:
                        e.gimbal_locked = False
                        gim_notes.append("unlocked a locked gimbal")
                    if GIMBAL_LIMIT is not None:
                        e.gimbal_limit = GIMBAL_LIMIT
                    gim_notes.append("range %.1f deg, limit %.2f"
                                     % (e.gimbal_range, e.gimbal_limit))
                except Exception as exc:
                    gim_notes.append("!! %s: %s"
                                     % (type(exc).__name__, exc))
        except Exception as exc:
            gim_notes.append("!! could not read engines: %s" % exc)
        self.n_gimbals = n_gim
        self.has_gimbal = n_gim > 0

        print("  gimballed engines: %d" % n_gim)
        for note in gim_notes:
            print("    %s" % note)
        if self.has_gimbal:
            print("  thrust vectoring available -- authority no longer depends")
            print("  on dynamic pressure, so steering stays live to %.0f Pa"
                  % FREEZE_STEER_Q_GIMBAL)
            print("  ...WHILE THE ENGINE IS BURNING.  On the way down it is")
            print("  fins only, and those need %.0f Pa." % CONTROL_MIN_Q)
        if n_gim == 0:
            print("  !! No engine gimbal.  The fins are this vehicle's ONLY")
            print("     control, and they do nothing until it is moving fast")
            print("     enough for them to bite.  A gimballing engine is the")
            print("     single biggest upgrade you can make to controllability.")

        # --- mission state ----------------------------------------------------
        self.state = self.PRELAUNCH
        self.t0 = self.s_ut()
        self.strikes = 0
        self.last_err = None
        self.last_err_t = None
        self._last_flight_log = 0.0
        self._departure_since = None
        self._last_rate_log = 0.0
        self._cda_seen = False
        self._alpha_ref = None
        self._inertia_ref = None
        self._alpha_filt = None
        self._alpha_raw = 0.0
        self._alpha_t = 0.0
        self._trim_filt = 0.0
        self._cr_prev = None
        self._cr_rate_b = 0.0
        self._cr_lead = 0.0
        self._meco_dn = 0.0
        self._meco_cr = None
        self._meco_met = None
        self._meco_speed = None
        self._meco_alt = None
        self._entry_spin = None
        self.flog = None        # set by main() when FLIGHT_LOG is on
        self.panel = None       # set by main() when USE_GAME_PANEL is on
        self._last_sol = None   # most recent solve(), for the panel readout
        self._outcome = None    # first outcome claimed wins
        self._cda_raw = INITIAL_CDA
        self._cda_dense_seen = False
        self._cda_lo = INITIAL_CDA
        self._cda_hi = INITIAL_CDA
        self._cda_win_t = 0.0
        self._cda_last_aligned = 0.0
        self._roll_sign = None
        self._roll_cal_start = None
        self._roll_cal_err0 = 0.0
        self._roll_cal_rate0 = 0.0
        self._roll_align_start = 0.0
        # Retry state for the roll-sign pulse.  See ROLL_CAL_RETRY.
        self._roll_cal_tries = 0
        self._roll_retry_start = None
        self._roll_retry_rate0 = 0.0
        self._roll_prev = None
        self._roll_d = 0.0
        self._term_dp = 0.0
        self._miss_prev = None
        self._aoa_peak = 0.0
        self._dn_rate = 0.0
        self._cr_rate = 0.0
        self._term_dy = 0.0
        # Throttle state.  Both latch on and never latch off -- see
        # throttle_schedule().
        self._boost_throttled = False
        self._cutoff_throttled = False
        self._throttle_cmd = 1.0
        self.launch_pos = self.s_pos()

        # Nominal great-circle range and bearing from the pad.
        self.range_angle_rad = central_angle(self.launch_pos, self.tgt_pos)
        self.range_total = self.R * self.range_angle_rad
        self.nominal_bearing = bearing_to(self.launch_pos, self.tgt_pos)

        # The direction the flight path runs AT THE TARGET, which is NOT the
        # bearing measured at the pad -- a great circle changes compass bearing
        # along its length.  The miss decomposition happens around the target,
        # so it must use the local direction there.
        self.approach_bearing = (bearing_to(self.tgt_pos, self.launch_pos)
                                 + 180.0) % 360.0
        print("  range to target: %.1f km   bearing %.1f deg  "
              "(%.1f deg at the target)"
              % (self.range_total / 1000.0, self.nominal_bearing,
                 self.approach_bearing))
        if self.range_total < 30000.0:
            print("  !! WARNING: target is only %.1f km away."
                  % (self.range_total / 1000.0))
            print("     This is a mortar shot, not a ballistic missile flight.")
            print("     The apogee will be inside the atmosphere, so there is")
            print("     no exo-atmospheric coast and drag dominates.  Expect")
            print("     poor accuracy.  Test at 100 km or more.")

        # --- ascent profile, scaled to the range ------------------------------
        # A minimum-energy shot peaks at roughly range/4 and the pitch-over has
        # to finish well before apogee -- call it range/12.
        # See DESIGN_NOTES "Ascent shaping".
        if AUTO_SCALE_ASCENT:
            self.pitch_end = clamp(self.range_total / 12.0, 300.0, PITCH_END_ALT)
            self.pitch_start = max(50.0, self.pitch_end * 0.05)
            # Kept deliberately low -- the kick must happen while Q is small.
            self.vert_alt = clamp(self.pitch_end * 0.03, 30.0, 250.0)
            self.vert_speed = clamp(VERTICAL_SPEED
                                    * math.sqrt(self.vert_alt / VERTICAL_ALT),
                                    10.0, VERTICAL_SPEED)
        else:
            self.pitch_end = PITCH_END_ALT
            self.pitch_start = PITCH_START_ALT
            self.vert_alt = VERTICAL_ALT
            self.vert_speed = VERTICAL_SPEED
        print("  ascent: vertical to %.0f m / %.0f m/s, pitch over %.0f-%.0f m"
              % (self.vert_alt, self.vert_speed, self.pitch_start, self.pitch_end))

        # Loft angle: minimum-energy elevation over a sphere is 45 - phi/4,
        # where phi is the angular range at the planet's centre.
        if LOFT_ANGLE is None:
            phi = math.degrees(central_angle(self.launch_pos, self.tgt_pos))
            self.loft = clamp(45.0 - phi / 4.0, LOFT_ANGLE_MIN, LOFT_ANGLE_MAX)
            print("  auto loft angle: %.1f deg (range angle %.1f deg)"
                  % (self.loft, phi))
        else:
            self.loft = float(LOFT_ANGLE)
            print("  loft angle: %.1f deg (manual)" % self.loft)

        # --- launch azimuth --------------------------------------------------
        # nominal_bearing stays pure great-circle geometry and is what the
        # crossrange decomposition is measured against.  launch_bearing is what
        # is actually flown: the same thing rotated to pre-cancel the planet's
        # rotation.  Biasing the geometry too would just move the target.
        self._azimuth_tof = None
        if AZIMUTH_BIAS is None:
            self.azimuth_bias, self._azimuth_tof = self.solve_launch_azimuth()
        else:
            self.azimuth_bias = float(AZIMUTH_BIAS)
        self.launch_bearing = (self.nominal_bearing + self.azimuth_bias) % 360.0
        if abs(self.azimuth_bias) > 1e-6:
            print("  launch azimuth: %.2f deg  (%.2f great circle %+.2f for "
                  "planetary rotation)"
                  % (self.launch_bearing, self.nominal_bearing,
                     self.azimuth_bias))
        else:
            print("  launch azimuth: %.2f deg (no rotation bias applied)"
                  % self.launch_bearing)

        self.bearing_cmd = self.launch_bearing

    # -- utility ---------------------------------------------------------------
    def met(self):
        return self.s_ut() - self.t0

    def log(self, msg):
        if VERBOSE:
            print("[%7.1f s] %-18s %s" % (self.met(), self.state, msg))

    def log_every(self, seconds, msg):
        """Log at most once per `seconds`.  The coast can last half an hour;
        without this it buries everything useful in scrollback."""
        now = self.s_ut()
        if now - self._last_rate_log >= seconds:
            self._last_rate_log = now
            self.log(msg)

    def goto(self, state):
        self.log("--> %s" % state)
        # Spin on entry to terminal is a strong predictor of the miss: the
        # 4.8 m flight arrived at 5.7 deg/s, the 426 m flight at 17.7.
        if state == self.TERMINAL and self._entry_spin is None:
            try:
                self._entry_spin = math.degrees(vmag(self.s_omega()))
            except Exception:
                pass
        self.state = state
        self.last_err = None

    # -- measurement -----------------------------------------------------------
    def update_drag_estimate(self):
        """Measure Cd*A directly from the game.

        drag force = Q * Cd*A, and kRPC gives us both the force vector and Q,
        so the effective drag area falls straight out.  Only trust it when
        there's meaningful dynamic pressure.
        """
        self.pred.mass = self.s_mass()
        q = self.s_q()
        if q <= 500.0:
            return                      # too little air to measure anything

        raw = vmag(self.s_drag()) / q
        self._cda_raw = raw
        now = self.s_ut()

        # Envelope over the last couple of seconds, so the log can show
        # whether the signal is converging or thrashing.
        if now - self._cda_win_t > 2.0:
            self._cda_lo, self._cda_hi, self._cda_win_t = raw, raw, now
        else:
            self._cda_lo = min(self._cda_lo, raw)
            self._cda_hi = max(self._cda_hi, raw)

        # Only believe the sample if the vehicle is pointing into the airflow,
        # which is what the predictor assumes for the rest of the flight.  If
        # nothing aligned turns up for CDA_STALE_TIMEOUT, take what we can get.
        aligned = (abs(self.s_aoa()) < CDA_MAX_SAMPLE_AOA
                   and abs(self.s_slip()) < CDA_MAX_SAMPLE_AOA)
        if aligned:
            self._cda_last_aligned = now
        elif CDA_ALIGNED_ONLY and (now - self._cda_last_aligned) < CDA_STALE_TIMEOUT:
            return                      # skip this one, wait for a clean look

        # Which regime is this sample telling us about?  A high, thin sample
        # says nothing about what the vehicle will feel at 50 kPa.
        #
        # AND IT HAS TO BE ON THE WAY DOWN: the vehicle passes through the
        # dense box twice and the ascent samples get there first, at half the
        # speed.  See DESIGN_NOTES "Which samples to believe".
        dense = (CDA_TWO_REGIME and self.s_alt() < CDA_DENSE_ALT
                 and q >= CDA_DENSE_MIN_Q
                 and vdot(self.s_pos(), self.s_vel()) < 0.0)
        if dense:
            self.pred.cda_dense = blend_cda(self.pred.cda_dense, raw,
                                            CDA_SMOOTHING, CDA_LOW_BIAS)
            self._cda_dense_seen = True
            return

        if self._cda_seen:
            self.pred.cda = blend_cda(self.pred.cda, raw,
                                      CDA_SMOOTHING, CDA_LOW_BIAS)
        else:
            self.pred.cda = raw
            self._cda_seen = True

    def solve(self, with_drag=True):
        """Run the predictor on the current state.

        Returns a dict of errors, or None if the prediction failed:
            downrange  : metres.  POSITIVE = falling SHORT of the target.
            bearing    : degrees, measured FROM THE LAUNCH POINT.  This is the
                         steering error during boost.  Positive = steer right.
            xrange_ok  : False when the bearing figure is not trustworthy yet.
            bearing_local : degrees, measured from the CURRENT position.  Only
                         meaningful in the terminal phase.
            miss       : total great-circle miss distance, metres.
            tof        : predicted time of flight remaining, seconds.
        """
        self.pred.use_drag = with_drag
        pos, vel = self.s_pos(), self.s_vel()
        result = self.pred.predict(pos, vel, self.tgt_alt)
        if result is None:
            return None
        impact, tof = result

        # Downrange and crossrange are BOTH measured as the offset of the
        # impact point from the TARGET, resolved along and across the nominal
        # flight direction.  Measuring instead from the MISSILE looks
        # equivalent and is not: as the missile converges on its own impact
        # point that distance goes to zero and the expression collapses to
        # "distance to target", blind to short versus long.

        # -- Crossrange -------------------------------------------------------
        # Measured from the LAUNCH POINT, not from where we currently are.
        # Early in the burn the predicted impact sits almost underneath the
        # missile, so the bearing from the missile to it is a 0/0.  A fixed
        # anchor is well-conditioned, and XRANGE_MIN_FRACTION gates the rest.
        d_from_launch = self.R * central_angle(self.launch_pos, impact)
        gate = max(500.0, XRANGE_MIN_FRACTION * self.range_total)
        if d_from_launch > gate:
            bearing = wrap180(self.nominal_bearing
                              - bearing_to(self.launch_pos, impact))
            xrange_ok = True
        else:
            bearing, xrange_ok = 0.0, False

        # Local bearing, for terminal guidance where we are close enough that
        # the geometry from the missile itself is the relevant one.
        if vmag(vsub(impact, pos)) > 200.0:
            bearing_local = wrap180(bearing_to(pos, self.tgt_pos)
                                    - bearing_to(pos, impact))
        else:
            bearing_local = 0.0

        # Split the target-to-impact offset along and across the nominal
        # flight direction.  b_off is which way the impact point lies from
        # the target, relative to the direction of flight:
        #   b_off near 0    -> impact is BEYOND the target      (long)
        #   b_off near 180  -> impact is short of the target    (short)
        #   b_off near +90  -> impact is to the right
        d_ti = self.R * central_angle(impact, self.tgt_pos)
        if d_ti > 0.5:
            b_off = wrap180(bearing_to(self.tgt_pos, impact)
                            - self.approach_bearing)
            crossrange = d_ti * math.sin(math.radians(b_off))
            # Positive = falling SHORT, matching the sign the rest of the
            # script uses, hence the leading minus.
            downrange = -d_ti * math.cos(math.radians(b_off))
        else:
            crossrange = 0.0
            downrange = 0.0

        # -- Project the crossrange forward to the target ---------------------
        # The raw figure above is the offset ACCUMULATED so far; the steering
        # loop wants the offset the current azimuth error will have produced by
        # the time the vehicle reaches the target's range.  Lateral offset from
        # an azimuth error e at central angle t is R sin(e) sin(t), so the two
        # differ by sin(t_target) / sin(t_impact).  A pure geometric rescaling,
        # not a prediction.  See XRANGE_PROJECT.
        crossrange_proj = crossrange
        if XRANGE_PROJECT:
            s_i = math.sin(central_angle(self.launch_pos, impact))
            s_t = math.sin(self.range_angle_rad)
            if s_i > 1e-6:
                crossrange_proj = crossrange * clamp(s_t / s_i,
                                                     XRANGE_PROJ_MIN,
                                                     XRANGE_PROJ_MAX)
            else:
                crossrange_proj = crossrange * XRANGE_PROJ_MAX

        lat, lon = vec_to_latlon(impact)
        self._last_sol = {
            "crossrange":    crossrange,
            "crossrange_proj": crossrange_proj,
            "downrange":     downrange,
            "impact_to_tgt": d_ti,
            "bearing":       bearing,
            "xrange_ok":     xrange_ok,
            "bearing_local": bearing_local,
            "miss":          self.R * central_angle(impact, self.tgt_pos),
            "tof":           tof,
            "lat":           lat,
            "lon":           lon,
        }
        return self._last_sol

    def cutoff_error(self, sol):
        """Downrange error the CUTOFF is trying to null, in metres.

        Positive = still falling short.  This is the raw predicted downrange
        plus AIM_OFFSET_DOWNRANGE, so that "zero" means "landing the offset
        distance BEYOND the target" -- see that setting for why aiming long
        is worth doing.

        Only the engine-cutoff decision uses this.  Everything after MECO
        aims at the real target.
        """
        return sol["downrange"] + AIM_OFFSET_DOWNRANGE

    def xrange_trim(self, sol):
        """Heading trim, in degrees, to null the predicted crossrange.

        Driven by METRES of crossrange rather than the bearing angle from the
        pad.  A fixed miss subtends a smaller and smaller angle as the range
        grows, so an angle-based gain quietly weakens on exactly the long
        shots where the miss is largest.

        Sign: crossrange positive means the impact lands RIGHT of the target,
        so the trim is negative and steers left.

        ONE SOURCE, for the whole burn.  Earlier versions switched from a
        launch-point bearing to metres of crossrange partway through, on the
        theory that crossrange could not see the error until the shot had
        developed.  It could -- the reading was simply scaled down by
        sin(central angle travelled).  The predictor now undoes that scaling
        (see XRANGE_PROJECT) and hands back crossrange_proj, which means the
        same thing at 80 s as it does at 108 s.  One metric, one gain, and no
        step change in the command halfway through the burn.
        """
        # Lead term: anticipate where the crossrange is heading rather
        # than reacting to where it is.  See BOOST_LEAD_T.
        now = self.s_ut()
        cr = sol["crossrange_proj"]
        if self._cr_prev is not None:
            dt = now - self._cr_prev[0]
            if 1e-3 < dt < 2.0:
                r = clamp((cr - self._cr_prev[1]) / dt,
                          -BOOST_RATE_MAX, BOOST_RATE_MAX)
                self._cr_rate_b += BOOST_RATE_SMOOTH * (r - self._cr_rate_b)
        self._cr_prev = (now, cr)
        # Only lead while the attitude is steady: a wobbling vehicle produces
        # a wobbling prediction, and leading on that feeds the oscillation
        # straight back into the steering.
        lead_t = BOOST_LEAD_T if self.pointing_error() < BOOST_LEAD_MAX_PT else 0.0
        self._cr_lead = cr + lead_t * self._cr_rate_b
        raw = -K_BOOST_XRANGE * self._cr_lead
        self._trim_filt += TRIM_SMOOTHING * (raw - self._trim_filt)
        return self._trim_filt

    def yaw_delivered(self):
        """How much of the requested heading trim survives the AoA clamp.

        If this is persistently smaller than the trim being asked for, the
        crossrange loop is being starved and no gain change will help.
        """
        vp, vh = self.velocity_pitch_heading()
        lim_y = max(self.aoa_limit(), ASCENT_YAW_AOA_FLOOR)
        return clamp(wrap180(self.bearing_cmd - vh), -lim_y, lim_y)

    def descent_mode(self):
        """Which of the three unpowered attitude regimes we are in.

            "powered"   - engine burning, none of this applies
            "hands off" - high and far from entry: write nothing at all and
                          let the vehicle drift.  No air, no damping, and
                          nothing worth pointing at yet.
            "align"     - descending towards the atmosphere, or in air still
                          too thin for the fins: point prograde, but on a hard
                          deflection cap so it is a slow deliberate slew.
            "control"   - enough air for the fins.  Normal guidance.

        The middle regime is the one added after the first hands-off flight.
        Drifting is right when there is nothing to hit; it stops being right
        about half a minute before the atmosphere arrives.
        """
        if self.powered():
            return "powered"
        q = HANDS_OFF_UNTIL_Q
        if q is None:
            q = self.control_min_q()
        if self.s_q() >= q:
            return "control"
        if (COAST_ALIGN
                and self.s_alt() < self.atmo_top + COAST_ALIGN_ABOVE
                and vdot(self.s_pos(), self.s_vel()) < 0.0):
            return "align"
        if not VACUUM_HANDS_OFF:
            return "control"
        return "hands off"

    def align_prograde(self):
        """Slow, hard-capped slew onto the velocity vector.

        The cap is the whole point.  Up here there is no aerodynamic damping,
        so the only thing stopping the loop building momentum it cannot get
        rid of is refusing it the authority to build it with.  boost=1.0
        overrides the thin-air cap opening, which would otherwise hand it full
        travel at exactly the wrong moment.
        """
        self.point_vec(self.velocity_unit(),
                       deflection=COAST_ALIGN_DEFLECTION, boost=1.0)

    def release_controls(self):
        """Hard zero on all three axes, and forget the controller's history.

        Zeroing matters more than just "no command": reaction wheels respond
        to whatever is written, so leaving a stale deflection in place would
        keep torquing the vehicle in vacuum for as long as the coast lasts.

        Clearing the history matters because the damping term differences
        against the previous cycle.  Coming back after several minutes of
        hands-off, that difference is meaningless -- the dt guard would reject
        it anyway, but resetting says so explicitly.
        """
        self.att.reset()
        try:
            c = self.vessel.control
            c.pitch = c.yaw = c.roll = 0.0
        except Exception:
            pass

    def powered(self):
        """Is the engine actually producing thrust right now?

        This matters far more than it looks.  A gimbal is only an actuator
        while the engine is burning -- the instant it shuts down, the vehicle
        that had authority at 40 Pa has fins and nothing else, and fins need
        thousands of Pa.  Everything that keys off "do we have thrust
        vectoring" has to ask this, not just whether a gimbal is fitted.
        """
        try:
            return self.vessel.thrust > 1.0
        except Exception:
            return False

    def control_min_q(self):
        """Dynamic pressure below which control authority starts collapsing.

        Gimbal figures while burning, fin figures once it is out.  Setting
        this once at startup from has_gimbal -- which is what the code used to
        do -- told the whole descent it could steer at 120 Pa on a vehicle
        whose engine had been off for four hundred seconds.
        """
        if self.has_gimbal and self.powered():
            return CONTROL_MIN_Q_GIMBAL
        return CONTROL_MIN_Q

    def freeze_steer_q(self):
        """Dynamic pressure below which steering commands are pointless."""
        if self.has_gimbal and self.powered():
            return FREEZE_STEER_Q_GIMBAL
        return FREEZE_STEER_Q

    def can_steer(self):
        """True when there is enough dynamic pressure to actually steer."""
        return self.s_q() >= self.freeze_steer_q()

    def steer(self, trim_deg, dt):
        """Apply a heading trim, rate-limited.

        The autopilot will happily try to follow a heading command that jumps
        30 degrees between cycles, which on a boosting rocket looks like the
        missile tumbling.  Limiting how fast the COMMAND can move keeps the
        vehicle stable even when the guidance solution is noisy.
        """
        want = self.launch_bearing + clamp(trim_deg, -MAX_YAW_TRIM, MAX_YAW_TRIM)
        step = MAX_HEADING_RATE * dt
        delta = clamp(wrap180(want - self.bearing_cmd), -step, step)
        self.bearing_cmd = (self.bearing_cmd + delta) % 360.0
        return self.bearing_cmd

    def velocity_pitch_heading(self):
        """Pitch and heading OF THE VELOCITY VECTOR, in degrees.

        Terminal guidance commands an angle relative to the airflow, not
        relative to the target, so we need to know where the airflow is.
        """
        pos, vel = self.s_pos(), self.s_vel()
        up, north, east = local_axes(pos)
        vu, vn, ve = vdot(vel, up), vdot(vel, north), vdot(vel, east)
        horiz = math.sqrt(vn * vn + ve * ve)
        pitch = math.degrees(math.atan2(vu, horiz))
        heading = math.degrees(math.atan2(ve, vn)) % 360.0
        return pitch, heading

    def aoa_limit(self):
        """How far off the velocity vector we are currently allowed to point.

        Tightens in proportion to dynamic pressure once past Q_SOFT_LIMIT,
        because that is where a few degrees of AoA turns into enough side load
        to snap fins off or flip the vehicle.
        """
        q = self.s_q()
        lim = MAX_ASCENT_AOA
        if q > Q_SOFT_LIMIT:
            lim *= Q_SOFT_LIMIT / q          # high Q: protect the airframe
        if q < self.control_min_q():
            # Low Q: fins stop working.  Irrelevant with a gimbal, which is
            # why control_min_q collapses when one is fitted.
            lim *= max(0.15, q / self.control_min_q())
            return lim                       # no floor here -- the floor is
                                             # what let it demand 8 degrees
                                             # at 300 Pa and thrash
        return max(lim, MIN_AOA_LIMIT)

    def throttle_schedule(self, sol):
        """Decide whether late boost has begun, and latch it.

        Latching matters.  The trigger quantity -- how close the predicted
        impact has come to the target -- is itself affected by throttling
        down, so an unlatched test would chatter across the threshold and
        modulate thrust at the frequency of the very oscillation we are
        trying to kill.  It arms once and stays armed.
        """
        if (self._boost_throttled or not ENGINE_CAN_THROTTLE
                or BOOST_THROTTLE_LOW >= 1.0 or sol is None):
            return
        if self.met() < BOOST_THROTTLE_MIN_MET:
            return
        if sol["impact_to_tgt"] < BOOST_THROTTLE_AT * self.range_total:
            self._boost_throttled = True
            self.log("THROTTLE DOWN to %.0f%% -- impact %0.0f km short, "
                     "slowing the plant for the endgame"
                     % (100.0 * BOOST_THROTTLE_LOW,
                        sol["impact_to_tgt"] / 1000.0))

    def manage_throttle(self):
        """The ONE place that writes vessel.control.throttle during boost.

        Three separate mechanisms want a say, and before this they fought:
        the Q limiter wrote 1.0 every cycle whenever Q was below the limit,
        which would have stamped straight over any other request.  They are
        combined here by taking the LOWEST demand, which is the safe way to
        merge throttle limits -- every one of them is a ceiling.

            1. MAX_Q limiter     - structural and control loads low down
            2. late-boost cutback - slows the guidance plant (BOOST_THROTTLE_LOW)
            3. cutoff approach    - fine range resolution (TERMINAL_THROTTLE)
        """
        if not ENGINE_CAN_THROTTLE:
            self._throttle_cmd = 1.0
            return
        cmd = 1.0
        if MAX_Q is not None:
            q = self.s_q()
            if q > MAX_Q:
                cmd = min(cmd, clamp(1.0 - (q - MAX_Q) / MAX_Q,
                                     MAX_Q_MIN_THROTTLE, 1.0))
        if self._boost_throttled:
            cmd = min(cmd, BOOST_THROTTLE_LOW)
        if self._cutoff_throttled:
            cmd = min(cmd, TERMINAL_THROTTLE)
        self._throttle_cmd = cmd
        self.vessel.control.throttle = cmd

    def command_attitude(self, want_pitch, want_heading, limit_heading=True):
        """Send an attitude to the autopilot, clamped to a safe angle of attack.

        The pitch program and the guidance laws both produce a DESIRED
        attitude.  This is the one place that decides whether the airframe can
        actually be asked for it.  Everything commands attitude through here.
        """
        vp, vh = self.velocity_pitch_heading()

        # No authority left: fly straight down the velocity vector.  A clean
        # thrust vector is worth more than a correction the vehicle cannot
        # execute.
        if self.s_q() < self.freeze_steer_q():
            self.point(vp, vh)
            return vp, vh

        # Separate allowances: pitch takes the tapered limit, yaw keeps a
        # floor.  See ASCENT_YAW_AOA_FLOOR.
        lim_p = self.aoa_limit()
        lim_y = max(lim_p, ASCENT_YAW_AOA_FLOOR)
        pitch = clamp(want_pitch, vp - lim_p, vp + lim_p)

        # Near-vertical the velocity heading is meaningless (a rocket going
        # straight up has no compass direction), so don't clamp against it.
        if limit_heading and vp < 80.0:
            delta = clamp(wrap180(want_heading - vh), -lim_y, lim_y)
            heading = (vh + delta) % 360.0
        else:
            heading = want_heading

        self.point(pitch, heading)
        return pitch, heading

    def control_alpha(self):
        """Angular acceleration per unit of full control input, rad/s^2.

        Averaged across pitch and yaw, and taking the weaker of the positive
        and negative directions so the figure is one the vehicle can always
        deliver.  Returns None if it cannot be measured.
        """
        try:
            moi = self.s_moi()
            tq = self.s_torque()
        except Exception:
            return None
        # kRPC axis order is (pitch, roll, yaw); available_torque is
        # (positive, negative).  Take the MEAN of the two directions, not the
        # smaller: the minimum tracks whichever direction has least authority
        # left at that instant, which swings with the control input rather
        # than describing the airframe.
        t_pitch = 0.5 * (abs(tq[0][0]) + abs(tq[1][0]))
        t_yaw = 0.5 * (abs(tq[0][2]) + abs(tq[1][2]))
        i_pitch, i_yaw = moi[0], moi[2]
        if i_pitch <= 0.0 or i_yaw <= 0.0:
            return None
        raw = 0.5 * (t_pitch / i_pitch + t_yaw / i_yaw)
        if raw <= 1e-9:
            return None

        # Heavy low-pass: inertia falls as the tanks drain, over a hundred
        # seconds.  Anything faster is measurement artefact.
        now = self.s_ut()
        # Powered: alpha follows inertia, which moves over ~100 s.
        # Unpowered: alpha follows dynamic pressure, which moves ten-fold in
        # eight.  See ALPHA_TAU_DESCENT.
        tau = ALPHA_TAU if self.powered() else ALPHA_TAU_DESCENT
        if self._alpha_filt is None:
            self._alpha_filt = raw
        else:
            dt = clamp(now - self._alpha_t, 0.0, 1.0)
            k = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
            self._alpha_filt += k * (raw - self._alpha_filt)
        self._alpha_t = now
        self._alpha_raw = raw
        return self._alpha_filt

    def control_inertia(self):
        """Mean of the pitch and yaw moments of inertia, kg m^2."""
        try:
            moi = self.s_moi()
        except Exception:
            return None
        i = 0.5 * (moi[0] + moi[2])
        return i if i > 0.0 else None

    def gain_scale(self):
        """How much to scale the attitude gains, given the vehicle right now.

        Below 1.0 softens the loop.  As the missile lightens it becomes more
        responsive to the same input, so the gains have to come down to keep
        the response -- natural frequency and damping both -- unchanged.
        """
        if not GAIN_NORMALISE:
            return 1.0
        basis = GAIN_BASIS
        if basis == "auto":
            # Inertia while the gimbal is the actuator, measured torque once
            # the fins are.  See GAIN_BASIS.
            basis = "inertia" if self.powered() else "torque"
        if basis == "inertia":
            if self._inertia_ref is None:
                return 1.0
            i = self.control_inertia()
            if i is None:
                return 1.0
            s = i / self._inertia_ref
        else:
            if self._alpha_ref is None:
                return 1.0
            a = self.control_alpha()
            if a is None:
                return 1.0
            s = self._alpha_ref / a

        # Gimbal torque is proportional to thrust, so dividing by the throttle
        # fraction restores the loop gain.  See GAIN_THROTTLE_COMP.
        if GAIN_THROTTLE_COMP and self._throttle_cmd > 0.05:
            s /= self._throttle_cmd
        return clamp(s, GAIN_SCALE_MIN, GAIN_SCALE_MAX)

    def capture_gain_reference(self):
        """Freeze the vehicle state the gains were tuned against."""
        self._alpha_ref = self.control_alpha()
        self._inertia_ref = self.control_inertia()
        if self._inertia_ref is None and self._alpha_ref is None:
            self.log("!! could not measure the vehicle; gains left as tuned")
            return
        self.log("gain reference (%s basis): inertia %.0f kg m^2, "
                 "alpha %.3f rad/s^2, mass %.0f kg"
                 % (GAIN_BASIS, self._inertia_ref or 0.0,
                    self._alpha_ref or 0.0, self.s_mass()))

    def deflection_boost(self):
        """How far to open or close the control-travel cap for the air we are in.

        Returns a multiplier on ATT_MAX_DEFLECTION.  Above 1.0 it opens the
        cap, below 1.0 it closes it.  Three regimes:

            thin  (q < ATT_DEFLECTION_Q)     open,  as ATT_DEFLECTION_Q / q
            tuned (between the two)          leave it alone
            dense (q > ATT_DEFLECTION_Q_HI)  close, as ATT_DEFLECTION_Q_HI / q

        Both tapers are the same idea: hold the control MOMENT roughly
        constant rather than the deflection angle, because the moment is what
        the airframe actually feels.  Only the opening half existed before,
        which is why reentry at 113 kPa was flown on a cap sized for 30 kPa.
        """
        q = max(self.s_q(), 1.0)
        if q < ATT_DEFLECTION_Q:
            return clamp(ATT_DEFLECTION_Q / q, 1.0,
                         1.0 / max(ATT_MAX_DEFLECTION, 1e-6))
        if q > ATT_DEFLECTION_Q_HI:
            return clamp(ATT_DEFLECTION_Q_HI / q,
                         ATT_DEFLECTION_MIN / max(ATT_MAX_DEFLECTION, 1e-6),
                         1.0)
        return 1.0

    def roll_rate(self):
        """Rotation rate about the vehicle's long axis, in degrees/second."""
        return math.degrees(vdot(self.s_omega_srf(), self.s_nose_srf()))

    def bank_angle(self):
        """How far the vehicle is rolled about its own nose, in degrees.

        Positive = rolled RIGHT (clockwise seen from behind).  Zero means the
        vehicle's belly points at the ground, so its pitch plane is vertical.

        WHY THIS IS WORTH AN RPC.  Nothing in the control laws needs it --
        steering is genuinely roll agnostic, because aim_offset builds its
        commands out of world vectors.  But the AoA and sideslip we log are
        BODY referenced, and without the bank angle they cannot be mapped
        onto range and crossrange.  On the flight that prompted this, a -30
        degree body-pitch excursion came out as 43 degrees of heading change
        and only 8 of flight path, which is only possible at about 80 degrees
        of bank -- and we had to infer that from the ratio of two other
        numbers rather than just reading it.

        Method: express local UP in the vessel's own axes (x right, y nose,
        z down).  Its right and up components give the bank angle directly.
        No handedness assumptions -- the same transform the attitude
        controller uses.
        """
        try:
            u = self.sc.transform_direction(
                (1.0, 0.0, 0.0),                      # up, in the surface frame
                self.vessel.surface_reference_frame,
                self.vessel.reference_frame)
        except Exception:
            return float("nan")
        return math.degrees(math.atan2(-u[0], -u[2]))

    def point(self, pitch, heading, deflection=1.0, kd_scale=1.0):
        """Aim the vehicle.  Every attitude command funnels through here.

        `deflection` scales the control-travel cap, so the terminal phase can
        be allowed the full range while the ascent stays gentle.

        See pitch_heading_to_direction() for why a bare direction is used
        rather than a pitch/heading pair.
        """
        self._want_dir = pitch_heading_to_direction(
            clamp(pitch, -90.0, 90.0), heading % 360.0)
        if USE_KRPC_AUTOPILOT:
            self.ap.target_direction = self._want_dir
            self.ap.target_roll = float("nan")
        else:
            self.att.roll_rate = self.roll_rate()
            self.att.deflection_boost = self.deflection_boost()
            self.att.gain_scale = self.gain_scale()
            self.att.update(self._want_dir, self.s_ut(), deflection, kd_scale)

    def miss_rates(self, sol, now):
        """How fast the downrange and crossrange miss are closing, m/s.

        Differentiated from the guidance solution itself, so the sign is
        correct by construction and no extra telemetry is needed.
        """
        dn, cr = sol["downrange"], sol["crossrange"]
        if self._miss_prev is not None:
            dt = now - self._miss_prev[0]
            if 1e-3 < dt < 2.0:
                a = TERMINAL_RATE_SMOOTH
                # Clamp before filtering: one absurd sample from a jumped
                # prediction would otherwise poison the filter for seconds.
                r_dn = clamp((dn - self._miss_prev[1]) / dt,
                             -TERMINAL_RATE_MAX, TERMINAL_RATE_MAX)
                r_cr = clamp((cr - self._miss_prev[2]) / dt,
                             -TERMINAL_RATE_MAX, TERMINAL_RATE_MAX)
                self._dn_rate += a * (r_dn - self._dn_rate)
                self._cr_rate += a * (r_cr - self._cr_rate)
        self._miss_prev = (now, dn, cr)
        return self._dn_rate, self._cr_rate

    def lead_errors(self, sol, now):
        """Miss values with the lead term folded in -- the zero-effort miss."""
        dn_rate, cr_rate = self.miss_rates(sol, now)
        t = clamp(sol["tof"], 0.0, TERMINAL_LEAD_MAX_T) * TERMINAL_LEAD
        return (sol["downrange"] + t * dn_rate,
                sol["crossrange"] + t * cr_rate,
                dn_rate, cr_rate)

    def tgo_gain(self, tof):
        """Gain multiplier from time-to-go.  See TERMINAL_TGO_REF."""
        tgo = max(tof, 0.5)
        return clamp((TERMINAL_TGO_REF / tgo) ** 2, 1.0, TERMINAL_TGO_GAIN_MAX)

    def velocity_unit(self):
        """Unit velocity in surface-frame components (x=up, y=north, z=east)."""
        pos, vel = self.s_pos(), self.s_vel()
        up, north, east = local_axes(pos)
        v = (vdot(vel, up), vdot(vel, north), vdot(vel, east))
        return vunit(v) if vmag(v) > 1.0 else (1.0, 0.0, 0.0)

    def point_vec(self, direction, deflection=1.0, kd_scale=1.0, boost=None,
                  trim_dir=None):
        """Aim at a raw unit direction in the surface frame.

        boost overrides the dynamic-pressure cap schedule.  Used by the
        pre-entry alignment, which needs a HARD cap in air thin enough that
        the schedule would otherwise open it all the way.
        """
        self._want_dir = direction
        if USE_KRPC_AUTOPILOT:
            self.ap.target_direction = direction
            self.ap.target_roll = float("nan")
        else:
            self.att.roll_rate = self.roll_rate()
            self.att.deflection_boost = (self.deflection_boost()
                                         if boost is None else boost)
            self.att.gain_scale = self.gain_scale()
            self.att.update(direction, self.s_ut(), deflection, kd_scale,
                            trim_dir)

    def aim_relative_to_velocity(self, dp, dy, deflection=1.0, kd_scale=1.0):
        """Point dp degrees above and dy degrees right of the velocity vector.

        Passes the velocity vector down as the TRIM REFERENCE, so the
        controller can work out how much angle of attack it is being asked to
        hold and feed that forward.  See ATT_AOA_FF -- without it the loop
        delivers about a tenth of what it is told at reentry dynamic
        pressures.
        """
        vhat = self.velocity_unit()
        self.point_vec(aim_offset(vhat, dp, dy), deflection, kd_scale,
                       trim_dir=vhat)

    def ballistic_shot(self, bearing):
        """Fire a drag-free arc from the pad on `bearing` at the loft angle,
        bisecting for the speed that reaches the TARGET'S RANGE.

        Returns (impact, tof, speed, reach) or None if no solution exists.

        The propagation is the same RK4 the guidance uses in flight, so it
        already carries the rotating-frame terms -- Coriolis included.  That
        is the whole reason this is usable for solving the launch azimuth:
        the answer comes from the same physics that will fly the mission.
        """
        up, north, east = local_axes(self.launch_pos)
        d = pitch_heading_to_direction(self.loft, bearing)
        aim = vadd(vscale(up, d[0]),
                   vadd(vscale(north, d[1]), vscale(east, d[2])))

        was_drag = self.pred.use_drag
        self.pred.use_drag = False
        best = None
        lo, hi = 500.0, 8000.0
        try:
            for _ in range(36):
                v = 0.5 * (lo + hi)
                r = self.pred.predict(self.launch_pos, vscale(aim, v),
                                      self.tgt_alt)
                if r is None:
                    hi = v          # never came down: too fast
                    continue
                impact, tof = r
                reach = self.R * central_angle(self.launch_pos, impact)
                best = (impact, tof, v, reach)
                if reach < self.range_total:
                    lo = v
                else:
                    hi = v
        except Exception:
            return None
        finally:
            self.pred.use_drag = was_drag

        if best is None:
            return None
        # Only trust it if the search actually converged on the target range.
        if abs(best[3] - self.range_total) > 0.02 * self.range_total:
            return None
        return best

    def delta_v(self):
        """Ideal delta-v left in the tanks, m/s.  None if unmeasurable.

        Rocket equation on live numbers from kRPC, so it tracks the vehicle
        rather than a constant someone typed in six rebuilds ago.  Vacuum Isp
        rather than current Isp because current reads zero with the engine
        off, which is when this gets called.
        """
        try:
            isp = self.vessel.vacuum_specific_impulse
            m0 = self.vessel.mass
            m_dry = self.vessel.dry_mass
        except Exception:
            return None
        if isp <= 0.0 or m_dry <= 0.0 or m0 <= m_dry:
            return None
        return isp * G0 * math.log(m0 / m_dry)

    def range_for_burnout_speed(self, v_burnout, loft):
        """Great-circle range reachable from a given burnout speed, metres.

        Reuses the pad-launch propagation that ballistic_shot() and the
        azimuth solver already rely on.  That model starts at sea level rather
        than at 50 km, so it needs more speed to fly the same distance --
        hence BURNOUT_SPEED_TO_PAD, which is measured rather than derived.
        Returns None if the arc never comes down.
        """
        up, north, east = local_axes(self.launch_pos)
        d = pitch_heading_to_direction(loft, self.launch_bearing)
        aim = vadd(vscale(up, d[0]),
                   vadd(vscale(north, d[1]), vscale(east, d[2])))
        was_drag = self.pred.use_drag
        self.pred.use_drag = False
        try:
            r = self.pred.predict(self.launch_pos,
                                  vscale(aim, v_burnout * BURNOUT_SPEED_TO_PAD),
                                  self.tgt_alt)
        except Exception:
            return None
        finally:
            self.pred.use_drag = was_drag
        if r is None:
            return None
        return self.R * central_angle(self.launch_pos, r[0])

    def range_envelope(self):
        """(min_range, max_range) in metres, or None if it cannot be worked out.

        MAX is delta-v limited: burn everything, sweep the loft angle, take
        whichever arc reaches furthest.

        MIN is cut-off limited: a ballistic missile cannot fly short by
        throttling, only by stopping early, and the earliest the state machine
        stops is the closed-loop handover.  The loft for a short shot is the
        auto value, which depends on the range -- so it is solved twice, which
        is plenty since the second pass barely moves.
        """
        dv = self.delta_v()
        if dv is None:
            return None
        v_max = dv * (1.0 - ASCENT_LOSS_FRACTION)
        if v_max <= 0.0:
            return None

        # -- maximum: best loft wins -----------------------------------------
        r_max = 0.0
        loft = LOFT_ANGLE_MIN
        while loft <= min(60.0, LOFT_ANGLE_MAX):
            r = self.range_for_burnout_speed(v_max, loft)
            if r is not None:
                r_max = max(r_max, r)
            loft += 2.5
        if r_max <= 0.0:
            return None

        # -- minimum: least burn the guidance will accept ---------------------
        v_min = min(MIN_BURNOUT_SPEED, v_max)
        r_min = None
        guess = 45.0                       # the short-range limit of the auto
        for _ in range(2):                 # formula; converges immediately
            r = self.range_for_burnout_speed(v_min, guess)
            if r is None:
                break
            r_min = r
            if LOFT_ANGLE is None:
                guess = clamp(45.0 - math.degrees(r / self.R) / 4.0,
                              LOFT_ANGLE_MIN, LOFT_ANGLE_MAX)
            else:
                guess = float(LOFT_ANGLE)
        if r_min is None or r_min >= r_max:
            return None
        return r_min, r_max

    def impact_crossrange(self, impact):
        """Crossrange of an impact point about the target, in metres.

        Same decomposition the in-flight solution uses: resolved around the
        direction the flight path runs AT THE TARGET, positive to the right.
        """
        d_ti = self.R * central_angle(impact, self.tgt_pos)
        if d_ti <= 0.5:
            return 0.0
        b_off = wrap180(bearing_to(self.tgt_pos, impact) - self.approach_bearing)
        return d_ti * math.sin(math.radians(b_off))

    def solve_launch_azimuth(self):
        """Work out how far off the great-circle bearing to launch.

        Fire a trial arc, see how far the planet's rotation carries it off
        target, rotate the launch azimuth to cancel that, repeat.

        The lever arm is R*sin(range angle), NOT the great-circle range: one
        radian of launch azimuth swings the impact point sideways by
        R*sin(theta), which on this shot is 536 km against a 664 km range.
        Using the range instead would under-correct by 20% every pass and
        converge slowly.

        Returns (bias_deg, tof) with bias 0.0 if it cannot solve.
        """
        lever = self.R * math.sin(self.range_angle_rad)
        if lever < 1000.0:
            return 0.0, None                 # target is at the antipode

        # NEVER return a bias that has not been flown in the propagator: the
        # residual measured on one pass belongs to the PREVIOUS bias.  So the
        # loop either exits having just verified the current value, or falls
        # through and verifies it explicitly before handing it back.
        bias = 0.0
        for _ in range(max(1, AZIMUTH_SOLVE_ITERS)):
            shot = self.ballistic_shot(self.nominal_bearing + bias)
            if shot is None:
                return 0.0, None
            cr = self.impact_crossrange(shot[0])
            if abs(cr) < AZIMUTH_SOLVE_TOL:
                return bias, shot[1]         # verified: this one lands on it
            # Impact right of target -> rotate the launch azimuth left.
            step = math.degrees(cr / lever)
            if abs(bias - step) > AZIMUTH_BIAS_MAX:
                print("  !! azimuth solver diverged (%.1f deg); flying the "
                      "raw great-circle bearing" % (bias - step))
                return 0.0, None
            bias -= step

        # Out of iterations.  Check what we are actually about to fly.
        shot = self.ballistic_shot(self.nominal_bearing + bias)
        if shot is None:
            return 0.0, None
        cr = self.impact_crossrange(shot[0])
        if abs(cr) > 5000.0:
            print("  !! azimuth solver did not converge (%.0f m residual); "
                  "flying the raw great-circle bearing" % cr)
            return 0.0, None
        return bias, shot[1]

    def estimate_flight_time(self):
        """Rough time of flight, worked out before launch.

        HONEST LIMITS: this ignores drag and the entire powered phase, so it
        is a vacuum approximation.  It is worth showing because it catches
        gross errors -- a target on the wrong continent, an absurd loft angle
        -- not because it predicts the real flight to any precision.  Returns
        None if no ballistic solution reaches the target at all, which is
        itself worth knowing before you light the engine.
        """
        shot = self.ballistic_shot(self.launch_bearing)
        return None if shot is None else shot[1]

    def launch_summary(self):
        """Print what is about to happen, and ask before doing it.

        Returns True to proceed, False to abort.
        """
        line = "=" * 62
        print("")
        print(line)
        print(" LAUNCH CHECK                                    v%s" % VERSION)
        print(line)
        print("   Target    : %s" % self.tgt_name)
        print("   Position  : %.5f N, %.5f E, elevation %.0f m"
              % (self.tgt_lat, self.tgt_lon, self.tgt_alt))
        print("   Range     : %.1f km on bearing %.1f deg"
              % (self.range_total / 1000.0, self.nominal_bearing))

        # Printed right under the range so the two numbers are read together,
        # before the Y/N prompt rather than after 120 seconds of burn.
        env = self.range_envelope()
        if env is not None:
            r_min, r_max = env
            dv = self.delta_v()
            print("   Envelope  : %.0f - %.0f km  (%.0f m/s delta-v, "
                  "+/-10%% estimate)"
                  % (r_min / 1000.0, r_max / 1000.0, dv or 0.0))
            if self.range_total > r_max:
                print("               !! TARGET IS %.0f km BEYOND MAX RANGE."
                      % ((self.range_total - r_max) / 1000.0))
                print("               !! Expect to run out of fuel short of "
                      "the cutoff solution.")
            elif self.range_total < r_min:
                print("               !! TARGET IS %.0f km INSIDE MIN RANGE."
                      % ((r_min - self.range_total) / 1000.0))
                print("               !! The engine cannot stop early enough; "
                      "expect to overshoot.")
            else:
                margin = 100.0 * (r_max - self.range_total) / r_max
                print("               target is inside, %.0f%% of max range "
                      "in reserve" % margin)
        if abs(self.azimuth_bias) > 1e-6:
            print("   Azimuth   : launching %.1f deg  (%+.2f deg for planetary"
                  % (self.launch_bearing, self.azimuth_bias))
            print("               rotation over the flight -- worth %.1f km "
                  "of crossrange)"
                  % (abs(self.R * math.sin(self.range_angle_rad)
                         * math.radians(self.azimuth_bias)) / 1000.0))
        else:
            print("   Azimuth   : launching %.1f deg (no rotation bias)"
                  % self.launch_bearing)
        print("   Loft      : %.1f deg%s"
              % (self.loft, " (auto)" if LOFT_ANGLE is None else ""))
        if abs(AIM_OFFSET_DOWNRANGE) > 1.0:
            print("   Aim       : cutoff biased %.1f km LONG; terminal dives"
                  % (AIM_OFFSET_DOWNRANGE / 1000.0))
            print("               back onto the target (short is the error it")
            print("               cannot fix, long is the one it can)")

        # The azimuth solver already propagated the corrected arc, so reuse
        # its answer rather than paying for another bisection.
        tof = (self._azimuth_tof if self._azimuth_tof is not None
               else self.estimate_flight_time())
        if tof is not None:
            print("   Flight    : ~%d min %02d s  (vacuum estimate; the real"
                  % (int(tof // 60), int(tof % 60)))
            print("               flight will differ once drag is included)")

        if ENGINE_CAN_THROTTLE and BOOST_THROTTLE_LOW < 1.0:
            print("   Throttle  : %.0f%% late boost, %.0f%% for cutoff"
                  % (100.0 * BOOST_THROTTLE_LOW, 100.0 * TERMINAL_THROTTLE))
        elif ENGINE_CAN_THROTTLE:
            print("   Throttle  : full, %.0f%% for cutoff only"
                  % (100.0 * TERMINAL_THROTTLE))
        else:
            print("   Throttle  : fixed (ENGINE_CAN_THROTTLE off)")

        print("   Vehicle   : %s" % self.vessel.name)
        print("               %d gimballed engine%s, %d control surface%s"
              % (self.n_gimbals, "" if self.n_gimbals == 1 else "s",
                 self.n_surfaces, "" if self.n_surfaces == 1 else "s"))
        if not self.has_gimbal:
            print("               !! no thrust vectoring -- steering will be")
            print("                  lost in the last seconds of burn")
        print(line)

        if not CONFIRM_LAUNCH or self.panel is not None:
            # The panel runs its own arm-and-confirm step, so asking here too
            # would mean answering the same question twice.
            return True
        try:
            answer = input(" Launch?  [Y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n Aborted.")
            return False
        if answer.startswith("y"):
            print("")
            return True
        print(" Aborted -- nothing was fired.")
        return False

    def roll_error(self):
        """Degrees of roll needed to put the target in the vehicle pitch plane.

        Take the horizontal unit vector pointing down the target bearing,
        express it in the vessel's own axes (x = right, y = nose, z = down),
        and ask how far round it sits from straight-below-the-nose.  Zero
        means the target lies in the y-z plane, which is the plane the
        vehicle pitches in -- so the whole turn can then be done in pitch.
        """
        b_surface = pitch_heading_to_direction(0.0, self.launch_bearing)
        b = self.sc.transform_direction(b_surface,
                                        self.vessel.surface_reference_frame,
                                        self.vessel.reference_frame)
        e = math.degrees(math.atan2(b[0], b[2]))
        # TWO rolls put the target in the pitch plane, 180 apart: target below
        # the nose, or above it.  Either is fine.  Taking whichever is nearer
        # folds the error into +/-90 and keeps it away from the +/-180
        # wrap-around, where it would flip sign on noise.
        alt_e = wrap180(e + 180.0)
        return e if abs(e) <= abs(alt_e) else alt_e

    def engage_control(self):
        """Take attitude control, using whichever controller is selected."""
        if USE_KRPC_AUTOPILOT:
            self.ap.engage()
        else:
            # Make sure nothing else is fighting us for the control axes.
            self.ap.disengage()
            self.vessel.control.sas = False
        print("  attitude control: %s"
              % ("kRPC autopilot" if USE_KRPC_AUTOPILOT
                 else "built-in PD (roll %s)" % ROLL_MODE))

    def release_control(self):
        """Hand the controls back and centre them."""
        try:
            self.ap.disengage()
        except Exception:
            pass
        if not USE_KRPC_AUTOPILOT:
            try:
                c = self.vessel.control
                c.pitch = c.yaw = c.roll = 0.0
            except Exception:
                pass

    def pointing_error(self):
        """Total pointing error in degrees, whichever controller is running."""
        if USE_KRPC_AUTOPILOT:
            try:
                return self.ap.error
            except Exception:
                return float("nan")
        return self.att.error

    def flight_log(self, extra="", interval=1.0):
        """One-line airframe diagnostic, about once a second.

        Read it like this:
          * control inputs near +/-1.00 with a big AP error   -> the missile
            physically cannot hold the commanded attitude.  That is an
            AIRFRAME problem: move fins aft, add fins, or lower MAX_ASCENT_AOA.
          * control inputs oscillating between the rails but AP error small
            -> the autopilot gains are too hot.  Raise AP_STOPPING_TIME.
          * control inputs small and calm -> attitude control is fine, and any
            miss is a guidance problem instead.
        """
        if not VERBOSE:
            return
        now = self.s_ut()
        if now - self._last_flight_log < interval:
            return
        self._last_flight_log = now
        ap_err = self.pointing_error()
        vp, vh = self.velocity_pitch_heading()
        # Total body rotation rate.  Above ~20 deg/s the vehicle is spinning,
        # whatever the individual axes say.
        spin = math.degrees(vmag(self.s_omega()))
        a = self.control_alpha()
        # What the damper is actually seeing, per axis.  Read next to "ctl":
        # ringing with these near zero means the damper is not seeing the
        # motion; large here with ctl saturating means authority-limited.
        if USE_KRPC_AUTOPILOT:
            rate = ""
        else:
            rate = " | rate p%+6.1f y%+6.1f d/s" % (self.att.body_rate_pitch,
                                                    self.att.body_rate_yaw)
        print("        alt %6.0f m  v %5.0f m/s  Q %6.0f Pa | vel pitch %5.1f "
              "hdg %5.1f | AoA %5.1f | spin %5.1f roll %+5.1f d/s "
              "bank %+4.0f%s | pt err %5.1f | %4.1ft a%.2f/%.2f g%.2f%s | "
              "ctl p%+.2f y%+.2f r%+.2f %s"
              % (self.s_alt(), self.s_spd(), self.s_q(), vp, vh, self.s_aoa(),
                 spin, self.roll_rate(), self.bank_angle(),
                 "" if self.att.roll_sign_known else "?",
                 ap_err,
                 self.s_mass() / 1000.0, a if a is not None else 0.0,
                 self._alpha_raw, self.gain_scale(), rate,
                 self.s_cpitch(), self.s_cyaw(), self.s_croll(), extra))

    def power_fraction(self):
        """Remaining electric charge as a fraction of capacity, 0..1."""
        try:
            r = self.vessel.resources
            cap = r.max("ElectricCharge")
            return r.amount("ElectricCharge") / cap if cap > 0 else 1.0
        except Exception:
            return 1.0

    def check_departure(self):
        """Detect loss of controlled flight and shut the motor down.

        Returns True if we have departed.  A healthy rocket keeps its nose
        within a few degrees of the airflow all the way up; sustained large
        incidence means the aerodynamics have won.

        TOTAL incidence, not pitch alone.  This used to test only
        abs(s_aoa()), which is the angle in the vehicle's PITCH plane -- so a
        departure that happened to land in yaw was completely invisible to
        it.  With the bank angle uncontrolled, which body axis an excursion
        falls into is arbitrary, so testing one of the two was a coin flip.
        """
        if not ABORT_ON_DEPARTURE:
            return False
        now = self.s_ut()
        incidence = math.sqrt(self.s_aoa() ** 2 + self.s_slip() ** 2)
        if incidence > DEPARTURE_AOA:
            if self._departure_since is None:
                self._departure_since = now
            elif now - self._departure_since > DEPARTURE_TIME:
                print("")
                print("!" * 62)
                print(" DEPARTURE FROM CONTROLLED FLIGHT")
                print("   Incidence held above %.0f deg for %.1f s at %.0f m, "
                      "Q %.0f Pa." % (DEPARTURE_AOA,
                                      now - self._departure_since,
                                      self.s_alt(), self.s_q()))
                print("   (AoA %+.0f, sideslip %+.0f, bank %+.0f)"
                      % (self.s_aoa(), self.s_slip(), self.bank_angle()))
                print("   The airframe is aerodynamically unstable, not the")
                print("   guidance.  In the VAB, centre of lift must sit BEHIND")
                print("   centre of mass, and must stay behind as fuel drains.")
                print("   Move fins aft / add fin area / add nose ballast.")
                print("!" * 62)
                self.abort("departure",
                           "incidence %.0f deg at %.0f m, Q %.0f Pa"
                           % (incidence, self.s_alt(), self.s_q()))
                return True
        else:
            self._departure_since = None
        return False

    def check_divergence(self, sol):
        if sol is None:
            self.strikes += 1
        elif not math.isfinite(sol["miss"]):
            self.strikes += 1
        else:
            self.strikes = 0
        if ABORT_ON_DIVERGENCE and self.strikes >= DIVERGENCE_STRIKES:
            self.log("predictor diverged %d cycles in a row" % self.strikes)
            self.abort("divergence",
                       "%d consecutive failed predictions" % self.strikes)
            return True
        return False

    def close_streams(self):
        """Release the kRPC streams so this object can be thrown away.

        Only needed when the panel's BACK button rebuilds Guidance for a
        different target; without it every retry leaks a dozen streams.
        Relies on the s_* naming every telemetry stream in __init__ uses.
        """
        for name, value in list(vars(self).items()):
            if name.startswith("s_") and hasattr(value, "remove"):
                try:
                    value.remove()
                except Exception:
                    pass

    def record_meco(self, sol):
        """Freeze the boost-phase result.  Boost sets the floor on the miss,
        so these are the numbers worth comparing across flights."""
        self._meco_dn = sol["downrange"]
        self._meco_cr = sol["crossrange"]
        self._meco_met = self.met()
        self._meco_speed = self.s_spd()
        self._meco_alt = self.s_alt()
        self.log_record(meco_met_s=round(self._meco_met, 1),
                        meco_downrange_m=round(self._meco_dn, 1),
                        meco_crossrange_m=round(self._meco_cr, 1),
                        meco_speed_ms=round(self._meco_speed, 1),
                        meco_alt_m=round(self._meco_alt, 1))

    def log_record(self, **fields):
        """Pass numbers to the flight log if one is running."""
        if self.flog is not None:
            self.flog.record(**fields)

    def log_finish(self, outcome, note=""):
        """Claim the outcome of this flight.

        The first claim wins, and the log is not actually written until
        close_log() -- so an abort path can say why the flight ended without
        cutting the transcript off before the shutdown it is about to do.
        """
        if self._outcome is None:
            self._outcome = (outcome, note)

    def close_log(self):
        """Write the flight log out.  Idempotent."""
        if self.flog is None:
            return
        outcome, note = self._outcome or ("incomplete", "")
        try:
            self.flog.record(cda_thin=round(self.pred.cda, 4),
                             cda_dense=round(self.pred.cda_dense, 4),
                             entry_spin_dps=(None if self._entry_spin is None
                                             else round(self._entry_spin, 2)),
                             aoa_peak_deg=round(self._aoa_peak, 2),
                             flight_time_s=round(self.met(), 1))
        except Exception:
            pass          # a dead connection must not lose the transcript
        self.flog.finish(outcome, note)

    def abort(self, reason="abort", note=""):
        self.goto(self.ABORT)
        self.log_finish(reason, note)
        try:
            self.vessel.control.throttle = 0.0
            self.shutdown_engines()
            self.release_control()
            self.vessel.control.sas = True
        except Exception:
            pass

    def shutdown_engines(self):
        self.vessel.control.throttle = 0.0
        for e in self.vessel.parts.engines:
            if e.active:
                try:
                    e.active = False
                except Exception:
                    pass   # solids genuinely cannot be switched off

    # =========================================================================
    # STATE HANDLERS
    # =========================================================================

    def do_prelaunch(self):
        """Arm everything and light the motor."""
        sol = self.solve(with_drag=True)
        if sol is None:
            self.log("predictor returned nothing on the pad (expected)")
        self.engage_control()
        self.point(VERTICAL_PITCH, self.launch_bearing)
        self.vessel.control.throttle = 1.0
        self.vessel.control.rcs = False
        time.sleep(1.0)
        self.log("ignition")
        self.vessel.control.activate_next_stage()
        self.goto(self.VERTICAL)
        return RATE_OPEN_LOOP

    def do_vertical(self):
        """Straight up, open loop.  The predictor is meaningless down here --
        at 50 m/s the 'impact point' is basically the launch pad."""
        # Straight up: no AoA limiting needed, we are flying along our own
        # velocity vector by definition.
        self.manage_throttle()
        self.point(90.0, self.launch_bearing)
        self.flight_log()
        if self.s_srf() > self.vert_alt and self.s_spd() > self.vert_speed:
            # Tanks full, engine lit, and the calmest part of the flight --
            # the right moment to record what "as tuned" means.
            self.capture_gain_reference()
            # The state runs whenever the roll SIGN is needed -- for the
            # damper, for the bank manoeuvre, or both.
            need_roll = ROLL_ALIGN or ROLL_MODE == "damp"
            self.goto(self.ROLLALIGN if need_roll else self.KICK)
        return RATE_OPEN_LOOP

    def roll_calibration_tick(self):
        """Retry the roll-sign pulse once there is enough air to feel it.

        Called every cycle of the powered ascent.  Does nothing unless the
        sign is still unknown AND there is real dynamic pressure -- see
        ROLL_CAL_RETRY for why the first attempt, made at 1.5 kPa, could
        never have worked.

        The pulse itself is harmless.  Roll is uncontrolled at this point
        anyway, and the steering is roll agnostic, so a degree or two per
        second of induced roll costs nothing -- and the moment the sign is
        known the damper starts taking it back out.
        """
        if (not ROLL_CAL_RETRY
                or self._roll_sign is not None
                or self._roll_cal_tries >= ROLL_CAL_MAX_TRIES
                or self.s_q() < ROLL_CAL_MIN_Q):
            return
        now = self.s_ut()
        if self._roll_retry_start is None:
            self._roll_retry_start = now
            self._roll_retry_rate0 = self.roll_rate()
            self._roll_cal_tries += 1
            self.log("roll sign still unknown -- retrying at Q %.0f Pa "
                     "(attempt %d of %d)"
                     % (self.s_q(), self._roll_cal_tries, ROLL_CAL_MAX_TRIES))
        self.att.roll_override = ROLL_CAL_RETRY_INPUT
        if now - self._roll_retry_start < ROLL_CAL_TIME:
            return

        delta = self.roll_rate() - self._roll_retry_rate0
        self.att.roll_override = None
        self._roll_retry_start = None
        if abs(delta) < ROLL_CAL_MIN_RESPONSE:
            self.log("   pulse moved the rate %+.2f d/s -- still not enough"
                     % delta)
            if self._roll_cal_tries >= ROLL_CAL_MAX_TRIES:
                self.log("   giving up: this airframe has no roll authority. "
                         "Roll stays free.")
            return
        # A positive input produced a rate change of `delta`.  The damper
        # applies -sign * K * rate, so it is negative feedback when sign has
        # the same polarity as delta.
        self._roll_sign = 1.0 if delta > 0 else -1.0
        self.att.roll_sign = self._roll_sign
        self.att.roll_sign_known = True
        self.log("roll sign = %+.0f (pulse changed rate %+.1f d/s at Q %.0f Pa)"
                 % (self._roll_sign, delta, self.s_q()))
        self.log("   roll damping is now LIVE -- bank angle will be held near "
                 "wherever it currently sits")

    def do_roll_align(self):
        """Bank the missile so the target lies in its pitch plane.

        Runs in two parts:

        1. CALIBRATE.  Hold a fixed roll input for a second and watch which
           way the roll error moves.  This measures the sign of
           control.roll -> rotation directly, instead of trying to derive it
           from KSP's left-handed frames, which is not reliably possible.

        2. ALIGN.  Proportional roll control onto zero roll error.

        After this the pitch program turns the missile straight at the target
        instead of having to yaw its way round.
        """
        now = self.s_ut()
        err = self.roll_error()

        # -- part 1: work out which way roll goes ---------------------------
        # Measured against the roll RATE, not the alignment angle, because rate
        # is what the damper acts on.
        if self._roll_sign is None:
            if self._roll_cal_start is None:
                self._roll_cal_start = now
                self._roll_cal_rate0 = self.roll_rate()
                self.log("calibrating roll direction (rate now %+.1f d/s)"
                         % self._roll_cal_rate0)
            self.att.roll_override = ROLL_CAL_INPUT
            if now - self._roll_cal_start >= ROLL_CAL_TIME:
                delta = self.roll_rate() - self._roll_cal_rate0
                if abs(delta) < ROLL_CAL_MIN_RESPONSE:
                    # The pulse did nothing.  Expected here: this runs at
                    # about 1.5 kPa, below the pressure at which fins work.
                    # See ROLL_CAL_RETRY.
                    self.log("!! roll pulse changed the rate by only %.2f d/s "
                             "at Q %.0f Pa." % (delta, self.s_q()))
                    if ROLL_CAL_RETRY:
                        self.log("   Too little air to tell.  Will retry above "
                                 "%.0f Pa." % ROLL_CAL_MIN_Q)
                    else:
                        self.log("   No usable roll authority; roll left free.")
                    self.att.roll_override = None
                    self.goto(self.KICK)
                    return RATE_OPEN_LOOP
                # Sign convention as in roll_calibration_tick().
                self._roll_sign = 1.0 if delta > 0 else -1.0
                self.att.roll_sign = self._roll_sign
                self.att.roll_sign_known = True
                self.log("roll sign = %+.0f (pulse changed rate %+.1f d/s)"
                         % (self._roll_sign, delta))
                self._roll_align_start = now
                if not ROLL_ALIGN:
                    # Sign is all we needed.  No bank manoeuvre -- steering is
                    # roll-agnostic.  See ROLL_ALIGN.
                    self.att.roll_override = None
                    self.log("roll: damping rate to zero, angle left free")
                    self.goto(self.KICK)
                    return RATE_OPEN_LOOP
            self.flight_log("| roll cal, err %.1f" % err)
            return RATE_OPEN_LOOP

        # -- part 2: roll onto the target plane -----------------------------
        # A cascade, not a single PD -- roll angle is a double integrator.  The
        # derivative comes from the error signal itself so its sign is correct
        # by construction, lightly smoothed because differentiating a noisy
        # angle at 20 Hz is otherwise mostly noise.
        d_err = 0.0
        if self._roll_prev is not None:
            dt = now - self._roll_prev[0]
            if 1e-3 < dt < 1.0:
                raw = wrap180(err - self._roll_prev[1]) / dt
                d_err = self._roll_d + 0.5 * (raw - self._roll_d)
        self._roll_d = d_err
        self._roll_prev = (now, err)

        # Outer loop: how fast do we WANT the error to be shrinking?
        want_rate = clamp(-ROLL_ALIGN_GAIN * err, -ROLL_MAX_RATE, ROLL_MAX_RATE)
        # Inner loop: drive the actual error rate onto that.
        self.att.roll_override = clamp(
            -self._roll_sign * ROLL_RATE_GAIN * (want_rate - d_err),
            -ROLL_ALIGN_MAX, ROLL_ALIGN_MAX)
        self.point(VERTICAL_PITCH, self.launch_bearing)
        self.flight_log("| roll align err %+.1f | rate %+.1f want %+.1f d/s "
                        "| cmd %+.2f"
                        % (err, d_err, want_rate, self.att.roll_override))

        done = abs(err) < ROLL_ALIGN_TOLERANCE
        timeout = (now - self._roll_align_start) > ROLL_ALIGN_TIMEOUT
        if done or timeout:
            self.log("roll alignment %s (error %.1f deg)"
                     % ("complete" if done else "TIMED OUT", err))
            self.att.roll_override = None   # back to a hard zero on roll
            self.goto(self.KICK)
        return RATE_OPEN_LOOP

    def do_kick(self):
        """Nudge the vehicle off vertical to start the gravity turn.

        Hold a fixed angle of attack BELOW prograde until the velocity vector
        itself has tilted to KICK_END_PITCH.  Once the velocity vector is off
        vertical, gravity has leverage on it and takes over the turning.

        This is short and happens at low speed on purpose: a few degrees of
        AoA at 3 kPa is harmless, whereas the same command at 45 kPa is what
        tears fins off.
        """
        self.manage_throttle()
        vp, _ = self.velocity_pitch_heading()
        cmd_p, _ = self.command_attitude(vp - KICK_AOA, self.launch_bearing)
        self.flight_log("| kick to %.1f (vel %.1f)" % (KICK_END_PITCH, vp))

        if vp <= KICK_END_PITCH:
            self.log("kick complete, velocity vector at %.1f deg" % vp)
            self.goto(self.PITCHOVER)
        elif self.s_q() > KICK_MAX_Q:
            self.log("!! Q reached %.0f Pa before the turn started (%.1f deg)."
                     % (self.s_q(), vp))
            self.log("   Handing over to the pitch program anyway.  If the")
            self.log("   missile then flies straight up, this vehicle is too")
            self.log("   fast too low: reduce TWR or allow throttling.")
            self.goto(self.PITCHOVER)
        return RATE_OPEN_LOOP

    def do_pitchover(self):
        """Open-loop tilt from vertical onto the loft angle.

        Pitch is scheduled against ALTITUDE, not time, so the profile is the
        same regardless of thrust-to-weight.  Turn slowly: a fast pitch-over
        at high dynamic pressure will snap fins off or flip the vehicle.
        """
        alt = self.s_alt()
        f = clamp((alt - self.pitch_start) / (self.pitch_end - self.pitch_start),
                  0.0, 1.0)
        pitch = 90.0 + (self.loft - 90.0) * f
        # Q climbs through 10 kPa early in this phase, the first point in the
        # flight where a roll pulse can actually be measured.
        self.roll_calibration_tick()
        self.manage_throttle()
        self.update_drag_estimate()
        sol = self.solve(with_drag=True)

        # Steer crossrange HERE, not only in closed loop: down low the air is
        # thick and there is a long burn left, so a correction is cheap.
        vp_now, _ = self.velocity_pitch_heading()
        if (sol is not None and sol["xrange_ok"]
                and vp_now < YAW_LOOP_MAX_PITCH and self.can_steer()):
            self.steer(self.xrange_trim(sol), 1.0 / RATE_OPEN_LOOP)

        cmd_p, _ = self.command_attitude(pitch, self.bearing_cmd)
        capped = "  <-- AoA LIMITED" if abs(cmd_p - pitch) > 1.0 else ""
        self.flight_log("| want %.1f got %.1f (cap p%.1f y%.1f)%s"
                        % (pitch, cmd_p, self.aoa_limit(),
                           max(self.aoa_limit(), ASCENT_YAW_AOA_FLOOR), capped))

        if sol is not None:
            # Early in the burn the predicted impact is basically underneath
            # us, so the shortfall equals the whole range.  See
            # CLOSED_LOOP_MIN_FRACTION.
            if sol["downrange"] < (1.0 - CLOSED_LOOP_MIN_FRACTION) * self.range_total:
                self.goto(self.CLOSED_LOOP)
        return RATE_OPEN_LOOP

    def do_closed_loop(self):
        """Closed-loop boost.  Steer in YAW only; leave range to the cutoff."""
        self.update_drag_estimate()
        sol = self.solve(with_drag=True)
        if self.check_divergence(sol):
            return RATE_CLOSED_LOOP
        if sol is None:
            return RATE_CLOSED_LOOP

        # Crossrange -> heading trim.  Two gates before we touch the rudder:
        #   1. the solution has to be well-conditioned (xrange_ok), and
        #   2. the velocity vector has to be shallow enough that a heading
        #      command means something.  Near vertical, "turn left" just
        #      rolls the missile.
        vp, _ = self.velocity_pitch_heading()
        if sol["xrange_ok"] and vp < YAW_LOOP_MAX_PITCH and self.can_steer():
            self.steer(self.xrange_trim(sol), 1.0 / RATE_CLOSED_LOOP)
        elif self.can_steer():
            self.steer(0.0, 1.0 / RATE_CLOSED_LOOP)
        # Below FREEZE_STEER_Q: leave bearing_cmd exactly where it is.

        pitch = self.loft
        if not ENGINE_CAN_SHUTDOWN:
            # No cutoff available, so range is controlled by loft.  Positive
            # downrange error = falling short = flatten out.  Stays on the
            # LOFTED branch so the relationship is monotonic and cannot hunt.
            self.loft = clamp(self.loft - K_LOFT * self.cutoff_error(sol),
                              LOFT_ANGLE_MIN, LOFT_ANGLE_MAX)
            pitch = self.loft

        self.throttle_schedule(sol)
        self.manage_throttle()
        self.command_attitude(pitch, self.bearing_cmd)
        self.flight_log()

        if self.met() % 2.0 < 0.25:
            self.log("miss %7.0f | dn %+8.0f (cut@%+7.0f) | cr %+6.0f ->%+7.0f "
                     "(%+5.0f m/s -> %+7.0f)%s | trim %+5.1f/%+4.1f | "
                     "thr %3.0f%% | Q %5.0f%s | pt %3.1f | tof %4.0f s"
                     % (sol["miss"], sol["downrange"], self.cutoff_error(sol),
                        sol["crossrange"], sol["crossrange_proj"],
                        self._cr_rate_b, self._cr_lead,
                        " " if sol["xrange_ok"] else "?",
                        wrap180(self.bearing_cmd - self.launch_bearing),
                        self.yaw_delivered(),
                        100.0 * self._throttle_cmd,
                        self.s_q(),
                        " STEER" if self.can_steer() else " FROZEN",
                        self.pointing_error(), sol["tof"]))

        if ENGINE_CAN_SHUTDOWN:
            # Gate on DOWNRANGE ONLY.  Cutoff is irreversible and
            # time-critical; waiting for crossrange too would let one bad
            # steering solution sail past the cutoff point.  Yaw steering
            # carries on inside the cutoff watch anyway.
            if abs(self.cutoff_error(sol)) < CUTOFF_WATCH_ERROR:
                self.goto(self.CUTOFF)
        else:
            # Solid motor: nothing to do but hold the profile until burnout.
            if self.vessel.thrust < 1.0:
                self.log("burnout")
                self.goto(self.COAST)
        return RATE_CLOSED_LOOP

    def do_cutoff_watch(self):
        """Freeze the attitude and wait for the exact instant to cut thrust.

        Downrange range is hugely sensitive to burnout speed, so this single
        timing decision sets most of the accuracy.  Rather than hope to catch
        the zero crossing on a loop tick, we measure how fast the error is
        shrinking, extrapolate to zero, and sleep exactly that long.
        """
        self.update_drag_estimate()
        sol = self.solve(with_drag=True)
        if self.check_divergence(sol) or sol is None:
            return RATE_CUTOFF

        # Pitch is locked from here, so the only thing changing the solution
        # is accumulating velocity, which makes the zero crossing predictable.
        # Yaw keeps trimming gently.
        vp, _ = self.velocity_pitch_heading()
        if sol["xrange_ok"] and vp < YAW_LOOP_MAX_PITCH and self.can_steer():
            self.steer(self.xrange_trim(sol), 1.0 / RATE_CUTOFF)
        self.command_attitude(self.loft, self.bearing_cmd)
        self.flight_log()
        e, now = self.cutoff_error(sol), self.s_ut()
        if self.last_err is None:
            self.log("cutoff watch: error %+.0f m" % e)

        # Throttling down near the end makes every tick of timing error cost
        # less range.  Latched, so a noisy solution cannot throttle back up
        # mid-endgame and change the closing rate being extrapolated from.
        if not self._cutoff_throttled and abs(e) < CUTOFF_WATCH_ERROR * 0.25:
            self._cutoff_throttled = True
            self.log("THROTTLE DOWN to %.0f%% for cutoff -- error %+.0f m"
                     % (100.0 * TERMINAL_THROTTLE, e))
        self.manage_throttle()

        if e <= 0.0:                       # already gone long -- cut now
            self.record_meco(sol)
            self.log("cutoff (crossed): aim error %+.0f m -> predicted "
                     "landing %.0f m %s, crossrange %+.0f m"
                     % (e, abs(sol["downrange"]),
                        "SHORT" if sol["downrange"] > 0 else "LONG",
                        sol["crossrange"]))
            self.shutdown_engines()
            self.goto(self.COAST)
            return RATE_COAST

        if self.last_err is not None and now > self.last_err_t:
            rate = (e - self.last_err) / (now - self.last_err_t)   # m/s, negative
            if rate < -1.0:
                t_zero = e / -rate
                self.log("error %+8.0f m  closing %7.1f m/s  cut in %.3f s"
                         % (e, -rate, t_zero))
                if t_zero < 1.0 / RATE_CUTOFF:
                    time.sleep(max(0.0, t_zero))
                    self.shutdown_engines()
                    self.record_meco(sol)
                    self.log("MECO: aim error %+.0f m -> predicted landing "
                             "%.0f m %s, crossrange %+.0f m"
                             % (e, abs(sol["downrange"]),
                                "SHORT" if sol["downrange"] > 0 else "LONG",
                                sol["crossrange"]))
                    self.goto(self.COAST)
                    return RATE_COAST

        self.last_err, self.last_err_t = e, now

        if self.vessel.thrust < 1.0:       # ran out of fuel before the solution
            self.log("!! burnout before cutoff solution -- shot will fall short")
            self.goto(self.COAST)
        return RATE_CUTOFF

    def do_coast(self):
        """Above the atmosphere there is nothing to fly through, so the
        trajectory is exactly what the propagator says.  Point nose-first
        along the velocity vector so reentry is stable."""
        self.vessel.control.throttle = 0.0
        mode = self.descent_mode()
        if mode == "hands off":
            self.release_controls()
        elif mode == "align":
            self.align_prograde()
        else:
            self.point_vec(self.velocity_unit())

        sol = self.solve(with_drag=True)
        if sol is not None:
            self.log_every(COAST_LOG_INTERVAL,
                           "coasting: miss %.0f m | impact %.4f / %.4f | "
                           "%.0f s to go | power %.0f%%  (%s)"
                           % (sol["miss"], sol["lat"], sol["lon"], sol["tof"],
                              self.power_fraction() * 100.0, mode))
        # Attitude trace through the coast, at COAST_LOG_INTERVAL rather than
        # every second because the phase is eight minutes long.  The number
        # that matters up here is "spin" -- Q is nil, so AoA and pt err mean
        # nothing.  See DESIGN_NOTES "Phase transitions".
        self.flight_log("| %s" % mode.upper(), interval=COAST_LOG_INTERVAL)

        # Reentry begins when we drop back through the atmosphere interface
        # on the way DOWN (dot(r, v) < 0 means the radius is shrinking).
        if self.s_alt() < self.atmo_top and vdot(self.s_pos(), self.s_vel()) < 0.0:
            self.goto(self.REENTRY)
        return RATE_COAST

    def do_reentry(self):
        """Drag is building.  Keep predicting with a live drag estimate, but
        don't correct hard yet -- in thin air at high Mach the fins have
        almost no authority and you'd only bleed off speed."""
        self.update_drag_estimate()
        sol = self.solve(with_drag=True)
        if self.check_divergence(sol) or sol is None:
            return RATE_REENTRY

        vp, vh = self.velocity_pitch_heading()
        # Gentle trim only: quarter gain, quarter of the AoA limit.
        g = self.tgo_gain(sol["tof"])
        e_dn, e_cr, _, _ = self.lead_errors(sol, self.s_ut())
        mode = self.descent_mode()
        if mode == "hands off":
            # No air, no engine, no actuators.  Anything written here is
            # reaction-wheel torque with nothing to damp it.  Let the airframe
            # weathercock.  See VACUUM_HANDS_OFF.
            self.release_controls()
        elif mode == "align":
            # Thin air, but on the way in.  Get the nose forward on a hard
            # deflection cap.  See COAST_ALIGN.
            self.align_prograde()
        else:
            if self.s_q() < REENTRY_STEER_Q:
                # Too thin to fly in and too unstable to trust.  Hold prograde.
                dp = dy = 0.0
            else:
                dp = clamp(0.25 * K_TERM_PITCH * e_dn * g,
                           -AOA_MAX * 0.25, AOA_MAX * 0.25)
                dy = clamp(-0.25 * K_TERM_YAW * e_cr * g,
                           -AOA_MAX * 0.25, AOA_MAX * 0.25)
            self.aim_relative_to_velocity(dp, dy, kd_scale=TERMINAL_KD_SCALE)

        # raw = this instant.  lo..hi = the spread over the last 2 s.
        # used = what the predictor is actually flying on.
        # lo and hi far apart means the vehicle is not settling.
        self.log_every(1.0,
                       "reentry: miss %6.0f m | CdA raw %.2f thin %.2f "
                       "dense %.2f%s | AoA %+.0f slip %+.0f bank %+.0f | "
                       "Q %.0f Pa%s"
                       % (sol["miss"], self._cda_raw, self.pred.cda,
                          self.pred.cda_dense,
                          "" if self._cda_dense_seen else "?",
                          self.s_aoa(), self.s_slip(), self.bank_angle(),
                          self.s_q(),
                          "  (%s)" % mode if mode != "control"
                          else "  (holding prograde)"
                          if self.s_q() < REENTRY_STEER_Q else ""))
        self.flight_log()

        if vmag(vsub(self.tgt_pos, self.s_pos())) < TERMINAL_RANGE:
            self.goto(self.TERMINAL)
        return RATE_REENTRY

    def do_terminal(self):
        """Fins have real bite now.  Fly an angle of attack relative to the
        airflow, sized by the residual miss, and clamped hard.

        Pitching the nose UP relative to velocity generates lift that
        stretches the trajectory; nose down shortens it.  So a positive
        downrange error (falling short) commands positive AoA.
        """
        self.update_drag_estimate()
        sol = self.solve(with_drag=True)

        vp, vh = self.velocity_pitch_heading()
        if sol is not None:
            g = self.tgo_gain(sol["tof"])
            e_dn, e_cr, dn_rate, cr_rate = self.lead_errors(sol, self.s_ut())
            dp = clamp(K_TERM_PITCH * e_dn * g,
                       -TERMINAL_AOA_MAX_PITCH, TERMINAL_AOA_MAX_PITCH)
            # Impact to the RIGHT of the target means steer LEFT.
            dy = clamp(-K_TERM_YAW * e_cr * g,
                       -TERMINAL_AOA_MAX_YAW, TERMINAL_AOA_MAX_YAW)
            # Spend the total-angle budget on crossrange first: pitch gives
            # least and costs most, so it yields.
            if math.hypot(dp, dy) > TERMINAL_AOA_MAX_TOTAL:
                room = math.sqrt(max(0.0, TERMINAL_AOA_MAX_TOTAL ** 2 - dy * dy))
                dp = clamp(dp, -room, room)
            # Print commanded AND achieved, plus the raw control deflections.
            # See DESIGN_NOTES "Reading the terminal log".
            #
            # Track the largest angle the airframe has actually achieved, so
            # the impact summary can recommend a cap.  Two rules:
            #   1. MAGNITUDE, NOT PER-AXIS -- commanded is in the velocity
            #      frame, achieved in the body frame, and bank is uncontrolled,
            #      so only the vector magnitude means the same thing in both.
            #   2. ONLY WHILE TRACKING (AOA_PEAK_TRACKING_TOL) -- an angle
            #      reached by departing is not a capability.
            cmd_mag = math.hypot(self._term_dp, self._term_dy)
            ach_mag = math.hypot(self.s_aoa(), self.s_slip())
            if ach_mag <= cmd_mag * AOA_PEAK_TRACKING_TOL:
                self._aoa_peak = max(self._aoa_peak, ach_mag)
            cap = (abs(self._term_dp) >= TERMINAL_AOA_MAX_PITCH - 0.1
                   or abs(self._term_dy) >= TERMINAL_AOA_MAX_YAW - 0.1)
            rail = max(abs(self.s_cpitch()), abs(self.s_cyaw())) > 0.97
            asking_too_much = (cap and rail
                               and self._aoa_peak < TERMINAL_AOA_MAX_YAW * 0.7)
            note = ""
            if asking_too_much:
                note = ("  <-- COMMANDING MORE THAN THE AIRFRAME CAN DO "
                        "(peak achieved %.0f deg; lower TERMINAL_AOA_MAX)"
                        % self._aoa_peak)
            elif cap and not rail:
                note = "  <-- AoA CAP is limiting (raise TERMINAL_AOA_MAX)"
            elif rail:
                note = "  <-- CONTROLS ON THE RAIL"
            self.log_every(1.0,
                           "terminal: miss %5.0f | dn %+5.0f (%+4.0f m/s -> %+5.0f) "
                           "cr %+5.0f (%+4.0f m/s -> %+5.0f) | tgo %4.1f x%.1f "
                           "| AoA %+5.1f/%+5.1f slip %+5.1f/%+5.1f | "
                           "pt %4.1f g%.2f | ctl p%+.2f y%+.2f%s"
                           % (sol["miss"], sol["downrange"], dn_rate, e_dn,
                              sol["crossrange"], cr_rate, e_cr,
                              sol["tof"], g, self._term_dp, self.s_aoa(),
                              self._term_dy, self.s_slip(),
                              self.pointing_error(), self.gain_scale(),
                              self.s_cpitch(), self.s_cyaw(), note))
        else:
            # Predictor gave up (too close / too fast).  Fall back to aiming
            # the nose at the target and riding it in.
            pos = self.s_pos()
            up, north, east = local_axes(pos)
            d = vsub(self.tgt_pos, pos)
            horiz = vsub(d, vscale(up, vdot(d, up)))
            direct_pitch = math.degrees(math.atan2(vdot(d, up), vmag(horiz)))
            direct_head = bearing_to(pos, self.tgt_pos, up, north, east)
            dp = clamp(direct_pitch - vp, -TERMINAL_AOA_MAX, TERMINAL_AOA_MAX)
            dy = clamp(wrap180(direct_head - vh),
                       -TERMINAL_AOA_MAX, TERMINAL_AOA_MAX)

        # Full control travel on the way down: the deflection cap exists to
        # protect a boosting rocket at max Q, and neither applies now.  The
        # commanded angles are low-passed so the missile flies a trend rather
        # than every jitter in the impact solution.
        self._term_dp += TERMINAL_CMD_SMOOTHING * (dp - self._term_dp)
        self._term_dy += TERMINAL_CMD_SMOOTHING * (dy - self._term_dy)
        # Offsets applied perpendicular to the velocity vector, NOT as
        # pitch/heading -- see aim_offset().
        self.aim_relative_to_velocity(
            self._term_dp, self._term_dy,
            deflection=TERMINAL_MAX_DEFLECTION / ATT_MAX_DEFLECTION,
            kd_scale=TERMINAL_KD_SCALE)

        if self.s_srf() < 20.0:
            self.goto(self.IMPACT)
        return RATE_TERMINAL

    def do_impact(self):
        pos = self.s_pos()
        lat, lon = vec_to_latlon(pos)
        miss = self.R * central_angle(pos, self.tgt_pos)
        print("")
        print("=" * 62)
        print(" IMPACT")
        print("   aim point : %.5f N, %.5f E" % (self.tgt_lat, self.tgt_lon))
        print("   hit point : %.5f N, %.5f E" % (lat, lon))
        print("   miss      : %.1f m" % miss)
        print("   flight    : %.1f s" % self.met())
        if self._meco_cr is not None:
            print("   at MECO   : downrange %+.0f m, crossrange %+.0f m"
                  % (self._meco_dn, self._meco_cr))
            print("   -> boost sets the floor; terminal only cleans up")
        if self._aoa_peak > 0.0:
            print("   peak total angle FLOWN ON COMMAND in terminal: %.1f deg"
                  % self._aoa_peak)
            print("   -> the airframe can do at least this much; angles it")
            print("      reached while departing are deliberately excluded")
        print("=" * 62)
        self.log_record(miss_m=round(miss, 2),
                        impact_lat=round(lat, 6), impact_lon=round(lon, 6))
        self.log_finish("impact")
        if self.panel is not None:
            self.panel.finished("impact", miss)
        return None

    # =========================================================================

    def run(self):
        handlers = {
            self.PRELAUNCH:   self.do_prelaunch,
            self.VERTICAL:    self.do_vertical,
            self.ROLLALIGN:   self.do_roll_align,
            self.KICK:        self.do_kick,
            self.PITCHOVER:   self.do_pitchover,
            self.CLOSED_LOOP: self.do_closed_loop,
            self.CUTOFF:      self.do_cutoff_watch,
            self.COAST:       self.do_coast,
            self.REENTRY:     self.do_reentry,
            self.TERMINAL:    self.do_terminal,
            self.IMPACT:      self.do_impact,
        }
        try:
            while self.state not in (self.ABORT,):
                t_start = time.time()

                # Revert / quickload detection.  If universal time jumps
                # backwards, every stream we hold is stale.
                if self.s_ut() < self.t0 - 1.0:
                    print("\nUniversal time went backwards -- the game was "
                          "reverted or reloaded.\nStopping.  Restart the "
                          "script for the new flight.")
                    self.log_finish("reverted", "universal time went backwards")
                    break

                if self.panel is not None:
                    self.panel.update(self)
                    if self.panel.abort_requested():
                        print("\nABORT pressed on the in-game panel.")
                        self.abort("panel_abort",
                                   "operator pressed ABORT in state %s"
                                   % self.state)
                        break

                # Airframe health check during every powered phase.
                if self.state in (self.VERTICAL, self.ROLLALIGN, self.KICK,
                                  self.PITCHOVER, self.CLOSED_LOOP,
                                  self.CUTOFF):
                    if self.check_departure():
                        break

                rate = handlers[self.state]()
                if rate is None:
                    break
                elapsed = time.time() - t_start
                time.sleep(max(0.0, 1.0 / rate - elapsed))
        except KeyboardInterrupt:
            print("\nInterrupted by user -- aborting.")
            self.abort("interrupted")
        except Exception as exc:
            # Record the crash before it propagates, so the transcript and the
            # results row survive it.
            self.log_finish("error", "%s: %s" % (type(exc).__name__, exc))
            raise
        finally:
            # Any path that reached here without an outcome ended early.
            self.log_finish("incomplete", "loop exited in state %s"
                            % self.state)
            self.release_control()
            self.close_log()     # last, so the shutdown is in the transcript


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def run_predictor_test(g):
    """Print one prediction for the current vessel state and stop.

    Use this on a coasting suborbital vehicle: if the printed impact point
    matches where it actually lands, the propagator is trustworthy and every
    later stage is built on solid ground.
    """
    g.update_drag_estimate()
    print("")
    print("--- PREDICTOR TEST -------------------------------------------")
    print("  altitude %.0f m   speed %.0f m/s   mass %.0f kg"
          % (g.s_alt(), g.s_spd(), g.s_mass()))
    if g.s_spd() < 50.0:
        print("")
        print("  !! The vehicle is barely moving, so this test tells you")
        print("     nothing: the predicted impact point is just where you")
        print("     already are, and the 'miss' is the full range to target.")
        print("     Hand-fly a suborbital hop and run this during the coast.")
        print("")
    print("  measured Cd*A: %.4f m^2   (Q = %.0f Pa)" % (g.pred.cda, g.s_q()))
    for label, drag in (("vacuum only", False), ("with drag", True)):
        sol = g.solve(with_drag=drag)
        if sol is None:
            print("  %-12s : no impact inside %.0f s" % (label, MAX_FLIGHT_TIME))
        else:
            print("  %-12s : %.5f N, %.5f E   |   %.0f s   |   miss %.0f m"
                  % (label, sol["lat"], sol["lon"], sol["tof"], sol["miss"]))
    print("--------------------------------------------------------------")


def run_attitude_test(g):
    """Hold one fixed attitude and report how well the vehicle tracks it.

    This answers a single question: can the autopilot fly this airframe?
    If commanded and actual stay within a few degrees and the control inputs
    are small, attitude control is fine and any miss is a guidance problem.
    If the error grows and the inputs sit on the rails, the vehicle is the
    problem and no amount of guidance tuning will help.
    """
    print("")
    print("--- ATTITUDE TEST --------------------------------------------")
    print("  holding pitch %.1f  heading %.1f  for %.0f s"
          % (ATTITUDE_TEST_PITCH, ATTITUDE_TEST_HEADING, ATTITUDE_TEST_SECONDS))
    print("  (the engine is NOT lit -- hand-fly to altitude first if you")
    print("   want this tested under aerodynamic load)")

    # On the pad the vehicle is bolted down and cannot respond, so the only
    # thing this checks is that the CONTROLLER behaves.
    clamped = False
    try:
        sit = str(g.vessel.situation)
        clamped = ("pre_launch" in sit) or ("landed" in sit) or ("splashed" in sit)
    except Exception:
        pass
    if clamped:
        print("")
        print("  NOTE: the vehicle is on the ground, so it cannot actually")
        print("  move.  A large steady pointing error is EXPECTED here and")
        print("  is not a fault.  What this still proves:")
        print("    * spin stays near zero  -> no roll runaway")
        print("    * roll command stays 0  -> nothing can wind up")
        print("    * deflection = ATT_KP x error, and stays off the rails")
        print("  To test whether it actually converges, hand-fly to a few km")
        print("  and run this again in the air.")
    print("")
    g.engage_control()
    fl = g.vessel.flight(g.vessel.surface_reference_frame)
    t_end = time.time() + ATTITUDE_TEST_SECONDS
    worst, next_print = 0.0, 0.0
    # Control runs fast; printing stays at 1 Hz so the output is readable.
    while time.time() < t_end:
        g.point(ATTITUDE_TEST_PITCH, ATTITUDE_TEST_HEADING)
        err = g.pointing_error()
        if math.isfinite(err):
            worst = max(worst, abs(err))
        if time.time() > next_print:
            next_print = time.time() + 1.0
            # Body-axis errors are broken out separately: with roll free,
            # which axis an error lands on depends on how the vehicle happens
            # to be rolled.
            print("    pitch %6.1f (want %5.1f)  hdg %6.1f (want %5.1f) | "
                  "body err: pitch %+6.1f yaw %+6.1f (total %5.1f) | "
                  "spin %5.1f d/s | ctl p%+.2f y%+.2f r%+.2f"
                  % (fl.pitch, ATTITUDE_TEST_PITCH, fl.heading,
                     ATTITUDE_TEST_HEADING,
                     g.att.e_pitch, g.att.e_yaw, err,
                     math.degrees(vmag(g.s_omega())),
                     g.s_cpitch(), g.s_cyaw(), g.s_croll()))
        time.sleep(0.05)
    g.release_control()
    print("  worst pointing error: %.1f deg" % worst)
    if clamped:
        print("  (on the ground -- judge this on spin and roll, not on error)")
    else:
        print("  In flight: under ~5 deg and settling = healthy.  Oscillating")
        print("  = raise ATT_KD.  Never closing the error = raise ATT_KP.")
        print("  Inputs pinned at the deflection cap with the error growing")
        print("  = not enough control authority for this airframe.")
    print("--------------------------------------------------------------")


def open_flight_log(mode):
    """Start a flight log for this run, or return None if logging is off.

    A logging failure must never stop a flight, so this reports and carries on.
    """
    if not FLIGHT_LOG:
        return None
    if mode != "fly" and not FLIGHT_LOG_TEST_MODES:
        return None
    try:
        return FlightLog(mode)
    except Exception as exc:
        print("flight log: could not be opened (%s) -- flying unlogged" % exc)
        return None


def build_guidance(conn, panel, flog):
    """Pick a target, build Guidance for it, and get the launch confirmed.

    Returns the Guidance object, or None if the operator cancelled.

    With no panel this is the old path: build once, print the launch check,
    answer the Y/N prompt.  With a panel, the operator cycles targets first,
    and BACK from the arm screen rebuilds Guidance for a different one --
    which is why the old object's streams are released before it is dropped.
    Guidance bakes the target into its range, bearing, azimuth solve and
    ascent profile at construction, so re-targeting genuinely means rebuilding.
    """
    body = conn.space_center.active_vessel.orbit.body
    targets, index = None, 0

    while True:
        if panel is not None:
            if targets is None:
                targets = list_targets(conn.space_center, body)
                print("in-game panel: %d target%s found"
                      % (len(targets), "" if len(targets) == 1 else "s"))
            launch_pos = conn.space_center.active_vessel.position(
                body.reference_frame)
            index = panel.choose_target(targets, launch_pos,
                                        body.equatorial_radius, index)
            if index is None:
                print("\nCancelled from the in-game panel -- nothing was fired.")
                return None
            use_target(targets[index])

        g = Guidance(conn)
        g.flog = flog
        g.panel = panel
        if flog is not None:
            flog.record(
                target_lat=g.tgt_lat, target_lon=g.tgt_lon,
                range_km=round(g.range_total / 1000.0, 3),
                launch_bearing_deg=round(g.launch_bearing, 4),
                azimuth_bias_deg=round(g.azimuth_bias, 4),
                aim_offset_m=AIM_OFFSET_DOWNRANGE,
                k_term_pitch=K_TERM_PITCH, k_term_yaw=K_TERM_YAW,
                term_aoa_max_pitch=TERMINAL_AOA_MAX_PITCH,
                term_aoa_max_yaw=TERMINAL_AOA_MAX_YAW,
                term_aoa_max_total=TERMINAL_AOA_MAX_TOTAL,
                att_kp=ATT_KP, att_kd=ATT_KD, att_aoa_ff=ATT_AOA_FF,
                terminal_kd_scale=TERMINAL_KD_SCALE,
                gain_basis=GAIN_BASIS, max_q=MAX_Q)
            # Re-taken now the target is fixed, so settings.json describes the
            # shot that was flown rather than the one in the file.
            flog.snapshot_settings()

        if MODE != "fly":
            return g

        confirmed = g.launch_summary()
        if panel is None:
            return g if confirmed else None

        verdict = panel.arm(g)
        if verdict == "launch":
            return g
        g.close_streams()
        if verdict == "abort":
            print("\nCancelled from the in-game panel -- nothing was fired.")
            return None
        print("\nPicking a different target.")


def main():
    if krpc is None:
        raise SystemExit("kRPC is not installed.  Run: pip install krpc")
    modes = ("fly", "attitude_test", "predictor_test")
    if MODE not in modes:
        raise SystemExit("MODE must be one of %s, not %r"
                         % (sorted(modes), MODE))

    # Opened before connecting, so a connection failure is in the transcript.
    flog = open_flight_log(MODE)
    conn = None
    panel = None
    try:
        conn = krpc.connect(name=CONNECTION_NAME, address=RPC_ADDRESS,
                            rpc_port=RPC_PORT, stream_port=STREAM_PORT)

        if USE_GAME_PANEL and MODE == "fly":
            panel = MissionPanel(conn)
            if not panel.ok:
                panel = None      # it already said why; fall back to console

        g = build_guidance(conn, panel, flog)
        if g is None:
            if flog is not None:
                flog.finish("scrubbed", "cancelled from the in-game panel")
            return

        print("")
        print("=== MODE: %s ===" % MODE.upper())
        if MODE == "fly":
            if panel is not None:
                panel.start_flight()
            g.run()               # g.run() closes the log itself
        else:
            run = (run_attitude_test if MODE == "attitude_test"
                   else run_predictor_test)
            run(g)
            if flog is not None:
                flog.finish("test_complete")
    except Exception as exc:
        if flog is not None:
            flog.finish("error", "%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        if flog is not None:
            flog.finish("incomplete")
        if panel is not None:
            panel.remove()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
