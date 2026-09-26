# Shopping list – biped robot, version 2

Goal of this version: a robot that walks like a person without falling, running the same model that is trained in
simulation (`addons/PPO/examples/Biped`). The list builds on the existing structure (goBILDA channels, aluminum feet,
pelvis with 24.4 cm between hip centers).

## What changes compared to today's robot

| | Today | Version 2 | Why |
|---|---|---|---|
| Servos | 10 × RDS3115MG | **10 × STS3250 + 2 × STS3095 (hip roll)** (+1 spare of each) | RDS3115 is too weak to stand on one leg (1.47 vs ~3 N·m needed); Feetech STS servos report angle, load, current and temperature |
| Joints per leg | 5 | **6** – hip yaw added | Without hip yaw the robot cannot correct its heading or turn; today it spins uncontrolled |
| Thigh and shin | 1120 U-Channel, 128 g | **1121 Low-Side U-Channel, 66 g** | Saves ~250 g in the legs; same hole pattern |
| Controller | Arduino Nano | **Arduino Nano ESP32** | The model (~24k weights) does not fit in the Nano's 2 KB of RAM |
| Servo drivers | 4 × PCA9685 | **One bus servo adapter** | STS servos are daisy-chained on one serial line |
| Sensors | none | **IMU + 8 pressure sensors** | The model needs body tilt and foot contact |
| Power | bench supply / converter | **3S 3000 mAh battery** | Untethered walking; 12.6 V matches the 12 V STS servos |

## 1. Servos and mechanics

| # | Item | Qty | Checked specs | Notes |
|---|---|---|---|---|
| 1 | **Feetech STS3250** (12 V version; C001 with bracket or C002 without) | 11 | 50 kg·cm stall, 12.5 kg·cm rated, 0.133 s/60°, 9–12.6 V, 74.5 g, 45.2×24.7×35 mm, 12-bit encoder, feedback of angle/load/current/temperature | 10 + 1 spare |
| 2 | **Feetech STS3095** (12 V) | 3 | 95 kg·cm stall, ~22.5 kg·cm rated, ~0.32 s/60°, 12 V, 194.5 g, 30×65×48 mm, same serial protocol | The 2 hip-roll joints + 1 spare. See "Why a hybrid" below |
| 3 | Aluminum U brackets for the STS3215/STS3250 size (long + short) | 11 sets | Must fit 45.2×24.7×35 mm | **Check the fit before buying in quantity** – order one set first |
| 4 | Brackets for the STS3095 | 3 sets | Must fit 30×65×48 mm | A different frame size: STS3250 brackets do not fit |
| 5 | **goBILDA 1121-0007-0192** Low-Side U-Channel, 192 mm | 4 | 66 g, aluminum, same hole pattern as 1120 | Thighs and shins |
| 6 | **goBILDA 1123-series pattern plate, 1×1 hole, 43×48 mm** | 8 (+2) | Fits flush inside the 1121 channel | Servo mounting at the channel ends – see section 6 |
| 7 | goBILDA 1120-0012-0312 (existing pelvis) | 0 | – | **Keep**. Stiffness matters more than weight in the pelvis, and it holds the battery and controller |
| 8 | Ball bearing for the hip yaw joint (e.g. 608ZZ, 8 mm bore) + matching shaft/hub | 2 (+2) | – | **Important**: the bearing carries the leg's weight and bending load so the servo shaft doesn't carry them alone. Design the joint before buying |
| 9 | M4 screws (goBILDA standard) and M3 (servo brackets), lock nuts | kit | – | Some bracket holes may need drilling |
| 10 | Thin rubber for the soles, 1–2 mm | 2 × 25×12 cm | – | Bare aluminum slips on the floor |

### Why a hybrid (STS3095 only at the hip roll)

The hardest load is standing on one leg: the hip-roll servo of the standing leg holds the whole body sideways.

| Option | Robot mass | Hip-roll torque needed | Servo rated / stall | Verdict |
|---|---|---|---|---|
| 12 × STS3250 | ~2.41 kg | ~2.9 N·m | 1.23 / 4.90 N·m → 2.4× over rated, 59% of stall | Works for walking and short one-leg stands; long stands heat the hip |
| **10 × STS3250 + 2 × STS3095** | ~2.65 kg | ~3.2 N·m | 2.2 / 9.3 N·m → 1.45× over rated, 34% of stall | **Recommended**: large margin where it matters, little extra mass |
| 12 × STS3095 | ~3.9 kg | ~4.6 N·m | 2.2 / 9.3 N·m → 2.1× over rated | Not recommended: the extra 1.4 kg eats most of the extra torque, and the servo is 2.4× slower (186 vs 451 °/s), too slow to catch a fall at the knee and ankle |

The STS3095 is slow, but the hip roll moves little and slowly, so its speed is not a problem there.

## 2. Controller and communication

| # | Item | Qty | Checked specs | Notes |
|---|---|---|---|---|
| 11 | **Arduino Nano ESP32** | 1 (+1) | VIN 6–21 V per the datasheet; 3.3 V logic; 8 analog inputs | Powered directly from the battery (12.6 V full) through VIN |
| 12 | **Waveshare Bus Servo Adapter (A)** | 1 | Supports ST/SC (including STS); input 9–12.6 V; jumper A = UART control | Connects the ESP32 UART to the servo bus |

## 3. Sensors

| # | Item | Qty | Notes |
|---|---|---|---|
| 13 | **Adafruit BNO085** (IMU) | 1 | Center of the pelvis; I2C; outputs a fused orientation |
| 14 | STEMMA QT / Qwiic cable | 1 | |
| 15 | **FSR 402** | 10 | 4 per foot at the corners, + 2 spares |
| 16 | 10k resistors | 10 | Voltage divider for each FSR |
| 17 | Perfboard | 1 | |

## 4. Power

| # | Item | Qty | Notes |
|---|---|---|---|
| 18 | **LiPo 3S (11.1 V), 3000 mAh, 30C or more, XT60** | 2 | ~240 g. **Not 4S**: 16.8 V trips the servos' over-voltage protection (above 16 V) |
| 19 | LiPo balance charger | 1 | Required |
| 20 | Low-voltage alarm | 1 | |
| 21 | Bench power supply 12 V, 30 A | 1 | For work without the battery |
| 22 | Silicone wire 16 AWG (main) and 18 AWG (per leg) | 2 m of each color | |
| 23 | XT60 connectors | 4 pairs | |
| 24 | Main switch 30 A+ and fuse holder with a 30 A fuse | 1 each | 30 A (not 20 A) because of the two STS3095 |
| 25 | Extension cables for Feetech servos (5264-3P connector) | 6 | Check that the STS3095 uses the same connector |
| 26 | Heat shrink, zip ties | kit | |

**Wiring:** each leg is a separate chain of 6 servos. Feed power into each chain separately, at the top of the hip,
with 18 AWG wire, so the full current doesn't flow through the thin servo cables.

## 5. Checks: will it really work

Calculated for the recommended configuration: 1121 legs, 10 × STS3250 + 2 × STS3095, 3S 3000 mAh battery.

**Mass:** about **2.65 kg** (today 2.2 kg).

**Torque:** see the table in section 1. In the simulation the one-leg stand uses on average about 24% of the
STS3250 stall torque at the hip roll (peak 39%), so the sim is not torque-limited; the stronger hip-roll servo is
there for the real robot's heating during long one-leg stands.

**Current and battery:**
- In the simulation the servos draw about 6–10 A together on average while walking.
- 3000 mAh battery: about **15–25 minutes**.
- Peak: 10 × 2.4 A + 2 × 9.8 A ≈ 44 A. A 30C battery gives 90 A ✓. The 30 A fuse protects against a short and
  won't blow during normal walking.

**Voltages and logic:**
- 3S battery: 9.9–12.6 V, inside the STS range (9–12.6 V) ✓
- Nano ESP32: VIN 6–21 V ✓
- Adapter: 9–12.6 V ✓
- IMU and FSR: 3.3 V from the ESP32 ✓
- 3.3 V UART from the ESP32 to the adapter: **verify** in the Waveshare documentation before connecting (the logic
  level is not stated explicitly)

**Compute:** the model is roughly 70→128→128→12. An ESP32 runs it in under 2 ms, far below the 20 ms between
decisions ✓

## 6. Mounting the servos to the 1121 channels

Today the servo mount is screwed to a 1123-series 1×1 hole pattern plate (43×48 mm) at the end of the 1120 channel.
The same plate works with the 1121: goBILDA lists it as fitting flush inside the channel, on the same hole pattern.

The difference: the 1121 sides are half as tall, so a plate held only by the side holes at the very end of the
channel is less supported and can flex under the hip and knee loads. Recommended:
- Screw the plate to the **base (bottom face)** of the channel, not only to the sides, with at least 4 screws.
- Put the plate one hole in from the end, so there is channel material around it on all sides.
- Where loads are highest (hip roll, knee), use a second plate or a goBILDA bracket on the other face so the servo
  is clamped from both sides.
- Order one channel + two plates first and test the stiffness by hand before buying all of them.

## 7. Feet

**Today (version 2 keeps them):** a 1 mm aluminum plate, 250 × 120 mm (81 g), with a 1–2 mm rubber sole and 4 FSR
402 sensors at the corners. The ankle joint sits **50 mm from the heel**, so there are 200 mm of foot in front of
it and only 50 mm behind it. The simulation uses exactly these numbers (`robot.py`: `FOOT_LENGTH`, `FOOT_WIDTH`,
`ANKLE_FROM_HEEL`).

**Why the ankle position matters:** the body stands roughly above the ankle, so the distance from the ankle to the
heel is how far the center of mass can move backward before the robot tips over backward. With 50 mm behind the
ankle, a small push backward is enough; the policy compensates by leaning the whole body forward (the pelvis pitches
forward ~9° in the current training).

**For the next robot:** put the ankle **in the middle of the foot, or slightly behind it** (about 110–125 mm from the
heel on a 250 mm foot).

| | Ankle 50 mm from the heel (today) | Ankle in the middle |
|---|---|---|
| Room to tip backward | 50 mm | 125 mm (2.5×) |
| Room to tip forward | 200 mm | 125 mm (still plenty) |
| Smallest margin (what limits stability) | 50 mm | 125 mm |
| Upright standing | needs a forward lean | upright |

The support area of the foot doesn't change, only where the body stands on it, which makes the margins in both
directions equal. A forward toe helps a human push off when walking fast, but this robot walks slowly with flat
feet, so an even margin is worth more. Most small humanoid robots put the ankle near the middle of the foot.

**Also for the next robot:**
- **A stiffer plate:** 1 mm aluminum bends under the whole robot standing on one foot, especially at the long toe.
  Use 2 mm (+81 g per foot), or keep 1 mm and fold a 10 mm lip along the long edges (much stiffer, almost no extra
  mass).
- **The FSRs under the four corners**, between the plate and the rubber, so each one measures the load at its corner.
- **Rounded corners** on the plate, so a corner doesn't catch on the floor when the foot tilts.

Before building, the simulation can compare the two ankle positions: changing `ANKLE_FROM_HEEL` in `robot.py` and
running the push test (`Evaluate.py --push-test`) gives the push the robot survives in each direction.

## 8. Are the channels optimal?

**goBILDA's hole pattern is a real advantage**: available, convenient, and compatible with all goBILDA
accessories. Worth staying with it. But:
- **1120 (current) is stronger and heavier than needed for thigh and shin.** 128 g per segment, while the loads on a
  2.5 kg robot are small.
- **1121 Low-Side** has the same hole pattern and width, with lower sides. **Only 66 g**, half. That saves about
  250 g in the legs: less hip torque, and a lighter leg to swing on every step.
- **In the pelvis** keep the 1120: the battery and controller sit there, and torsional stiffness matters.
- **Length:** 192 mm for thigh and shin gives a near-human ratio (thigh ≈ shin) and a height of about 78 cm to the top of the pelvis with straight legs (with the hip yaw servo; ~76 cm standing
with the knees slightly bent), plus whatever sits on top of the pelvis. No reason
  to change it. A shorter leg would lower the center of mass and the loads, but also the step length.
- **Lighter alternative:** carbon tubes. Less convenient (no hole pattern, special adapters), not worth it at this
  stage.

## 9. From simulation to the real robot

A policy that is stable in the simulation will not automatically be stable on the real robot. To close the gap:
1. **Measure the real servos** (one STS3250 and one STS3095 on the bench): response delay, speed under load, and
   the angle error at a given load. Update `servos.py` with the measured numbers.
2. **Measure the real robot**: mass of each part and distances between the axes after assembly; update `robot.py`.
3. **Domain randomization in training** (to be added before the real-robot model): random mass (±15%), friction,
   servo strength (±20%), control delay (0–40 ms), sensor noise on the IMU and the angles. A policy that survives
   all of these is much more likely to work on the real robot.
4. **First tests with a safety harness** (a rope from above), starting with standing, then the weight shift, then
   one leg – the same order as the curriculum.
5. **Watch the servo temperature** reported by the STS servos; stop if a hip servo passes 60 °C.

## 10. What to check before the full order

1. **Order one STS3250 + one STS3095 + a bracket set for each + the adapter** first, and verify mechanical fit
   and communication with the ESP32.
2. **Design the hip yaw joint:** where the servo sits (in the pelvis, shaft down) and how the bearing carries the
   leg. Its dimensions change the height and must be updated in the simulation.
3. **Re-measure the distances between the axes** after assembly and update `robot.py`.

## What stays from the current robot

- The 1120 thigh and shin channels: not needed with the 1121, but usable for other parts.
- 10 RDS3115 servos and 4 PCA9685 boards: not used in version 2.
- Diymore converter: useful for a bench supply (24 V → 12 V) if there is no 12 V supply.
- The feet: they stay, with a rubber sole (see section 7 for the next robot).

## Sources

- [Feetech STS3250 – OpenELAB](https://openelab.io/products/feetech-sts3250-c002-servo-12v)
- [Feetech STS3250 – Feetech](https://www.feetechrc.com/en/562636.html)
- [Feetech 50 kg·cm TTL spec sheet (HLS3950M/STS3250)](https://ecksteinimg.de/Datasheet/Feetech/FT01021.pdf)
- [Feetech STS3095 – RCDrone](https://rcdrone.top/products/feetech-sts3095-servo-motor)
- [goBILDA 1121-0007-0192](https://www.gobilda.com/1121-series-low-side-u-channel-7-hole-192mm-length/)
- [goBILDA 1121 Series](https://www.gobilda.com/1121-series-low-side-u-channel)
- [goBILDA 1123 Series pattern plates](https://www.gobilda.com/1123-series-pattern-plates/)
- [Arduino Nano ESP32 datasheet (ABX00083)](https://docs.arduino.cc/resources/datasheets/ABX00083-datasheet.pdf)
- [Waveshare Bus Servo Adapter (A) – Wiki](https://www.waveshare.com/wiki/Bus_Servo_Adapter_(A))
- [Waveshare Bus Servo Adapter (A)](https://www.waveshare.com/bus-servo-adapter-a.htm)
- [Aluminum brackets for STS3215 – Amazon](https://www.amazon.com/RCmall-Aluminum-Bracket-Feetech-Feedback/dp/B0GVLSJ6TG)
