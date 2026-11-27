# BEV → Leaf DBW: two steps

Everything else is already done and tested off-car (`bash bev_bridge/tests/run_offline.sh`).

The ESP firmware (`dbw_esp_full`, `adapter_esp`) is held in the MiLa repos and is not part of this
one; step 1 flashes it from there. Its interface and limits are documented in
[`bev_bridge/README.md`](bev_bridge/README.md) and recorded in `bev_bridge/hardware.json`.

---

## Step 1 — flash both ESPs

**Ignition OFF.** Quiet CAN gives a clean bootloader handshake; the MCP2515 INT line corrupts
serial while the bus is live. Flash on the bench over laptop USB, then unplug the laptop.

```bash
CLI="/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli"
export ARDUINO_DIRECTORIES_DATA=$HOME/Library/Arduino15
FQBN=esp32:esp32:esp32
cd ~/Desktop/Mila_Develop        # the firmware lives in the MiLa repo, not in this one

"$CLI" board list                                  # find each board's /dev/cu.* port

# adapter ESP  (the one on USB to the Jetson)
"$CLI" compile --fqbn $FQBN firmware/adapter_esp
"$CLI" upload  --fqbn $FQBN -p /dev/cu.usbserial-XXXX firmware/adapter_esp

# DBW board  (the one on the car's CAN + the actuators)
"$CLI" compile --fqbn $FQBN firmware/dbw_esp_full
"$CLI" upload  --fqbn $FQBN -p /dev/cu.usbserial-YYYY firmware/dbw_esp_full
```

If an upload fails or the flash is corrupt, retry at `--upload-field speed=115200`.

### Boot lines you must see (115200 baud)

```
ADAPTER: MCP2515 OK
ADAPTER: ready. '<steer> [<speed> [<pedal> [<mode>]]]' over USB. 0x740 v2 @ 50Hz.

DBW-FULL: DAC1 OK DAC2 OK
DBW-FULL: MCP2515 OK
DBW-FULL: ready (steer angle + speed hold). needs 0x740 + 0x002; speed needs 0x284.
```

`DAC2 MISSING` → the DAC2 I2C wiring (GPIO33/32) has dropped out before; recheck it, don't drive.

### Quick sanity check

Ignition **ON**, watch the DBW board's status line:

```
DBW ... | 0x002:1234 0x284:5678
```

Both counters climbing = CAN is healthy and `dbw_esp_full` has the steering and wheel-speed
feedback it needs. This is the normal, proven state — the same firmware has driven many mila-av
paper runs. If they ever sit at `0`, the board is not receiving: that happened once, on 2026-10-26,
and the fallback is `dbw_esp_serial` + `JetsonCAN/jetson/dbw_bridge.sh` with `--device /tmp/dbw_bridge`.

### Power rule — non-negotiable

Once on the car, the **ESP is powered from the car, never from laptop USB**. A floating laptop ground
offsets every DAC-injected voltage: the EPS and ABS see implausible sensor pairs and latch steering
and brake faults, and the values reaching the ECUs are wrong, so the test means nothing either way.
Same for the DBW board over Bluetooth — power it from the **car's** USB port, not the Jetson.

---

## Step 2 — run your side

Hanchen runs his usual BEV stack and changes nothing. Everything on our side is one script:

```bash
cd /home/agxthor/SingleCameraBev
git fetch && git checkout bridge-actuation
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$PWD/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml

python3 bev_bridge/mila_run.py --preflight-only    # checks only, no ROS, no bridge
python3 bev_bridge/mila_run.py                     # + dry-run bridge + live status
python3 bev_bridge/mila_run.py --actuate --engage  # the real thing
```

`mila_run.py` does five things in order and stops at the first that fails:

1. **Serial preflight** — confirms the device in `hardware.json` is the **adapter**. Both boards are
   CP2102 serial `0001`, so the `/dev/serial/by-id` alias follows whichever enumerated last, and
   opening the DBW board instead would command nothing. Read-only, DTR/RTS held low so it does not
   reset the board, and it only counts **fresh** output so stale buffered bytes cannot pass the check.
2. **Topic preflight** — watches his `/camera_lidar_bev/path_tracking/status` and reports the rate
   plus every field the bridge gates on. A replay run, a missing `vehicle_transform_verified`, a
   stale stamp: it names which one, before the car is involved.
3. **Bridge** — starts `bev_dbw_bridge.py` as a child (dry-run unless `--actuate`). Deliberately not
   `run_all.sh`, which would launch a second copy of Hanchen's stack on top of his.
4. **Status** — one live line: gate state, commanded steer/speed, and the `0x741` telemetry coming
   back (measured angle, measured mph, engaged, age).
5. **Engage** — only with `--engage`, and only after you type `drive` at the prompt. It publishes the
   explicit release, then the heartbeat at 10 Hz.

It never engages on its own, and `--preflight-only` exits nonzero on a bad board so you can gate on
it. Ctrl-C releases, terminates the bridge (whose own shutdown brakes to a stop), and leaves the
firmware watchdog holding the board safe.

### What to watch

The status line carries the telemetry **measured** off the car, not what we commanded. If `age`
shows `none` or climbs, the board is not echoing `0x741` and the bridge latches `telemetry_lost`
within a second. A latched stop names its cause and does not resume on a new engage; restart after
fixing it. The causes to expect:

| latched cause | what it means |
|---|---|
| `speed_hold_not_responding` | speed commanded, car stationary — check the `0x284` counter |
| `overspeed` | measured above 6 + 2 mph; the firmware has no overspeed trip of its own |
| `telemetry_lost` | board or adapter stopped echoing `0x741` |
| `steering_not_tracking` | measured wheel far from commanded for 2 s |

Afterwards the per-tick telemetry is in the bridge's session directory, `events.jsonl` — that is the
record that settles the October speed question.

**The hardware e-stop / JD-VCC relay cut is the stop at any point.** Nothing here replaces it.

### If you need to drive the whole stack yourself

When Hanchen is *not* running his stack, `run_all.sh` brings up his sensors, localization and BEV
plus the bridge together. Its engage topic is `/bev_dbw/hardware_engage`, not `/bev_dbw/engage`:

```bash
bash bev_bridge/run_all.sh --dry-run
bash bev_bridge/run_all.sh --actuate --hardware-config bev_bridge/hardware.json --max-speed-mph 6
```

Do not run this and `mila_run.py` at the same time — that is two bridges and two stacks.
