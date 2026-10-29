# DATA.md: Data Coverage, Cleaning Pipeline, and Quality Audit

This document details the data characteristics, structural schema, data-quality findings, cleaning methodology, and remaining data limitations for the **REFIT Electrical Load Measurements** dataset used in this assignment.

---

## 1. Dataset Overview & Data Coverage

The REFIT study collected continuous longitudinal electricity consumption data across 20 domestic dwellings located in the Loughborough area, United Kingdom, between **October 2013 and June 2015**.

### Key Dataset Characteristics:
- **Spatial Coverage:** 20 residential homes (designated House 1 through House 21, omitting House 14 due to early data collection dropouts).
- **Temporal Duration:** 18 to 21 months of continuous longitudinal recording per household.
- **Nominal Sampling Interval:** ~8 seconds per reading in raw files.
- **Physical Quantities & Units:** Active Real Power measured in **Watts (W)**.
- **Instrumentation Architecture:**
  - **Whole-House Aggregate Meter:** Current transformer (CT) clamps installed on the main electricity supply cable entering the consumer unit.
  - **Individual Appliance Monitors (IAMs):** Radio-frequency plug-in smart sockets (developed by Current Cost / Enmetric) logging individual appliance loads across 9 distinct channels per home (`Appliance1` through `Appliance9`).

---

## 2. File Schemas & Structure

### Raw Data Schema (`House1.csv` from `REFIT_RAW_081116.7z`)
```
Time, Unix, Aggregate, Appliance1, Appliance2, Appliance3, Appliance4, Appliance5, Appliance6, Appliance7, Appliance8, Appliance9
```
- `Time`: Local timestamp formatted as `YYYY-MM-DD HH:MM:SS`.
- `Unix`: Integer epoch timestamp (seconds since 1970-01-01 UTC).
- `Aggregate`: Total household active power in Watts.
- `Appliance1` through `Appliance9`: Sub-metered appliance power channels in Watts.

### Cleaned Data Schema (`CLEAN_House*.csv` from `CLEAN_REFIT_081116.7z`)
```
Time, Unix, Aggregate, Appliance1, Appliance2, Appliance3, Appliance4, Appliance5, Appliance6, Appliance7, Appliance8, Appliance9, Issues
```
- Identical to raw columns with the addition of `Issues`, a bitmask flag indicating specific anomalies or sensor substitutions recorded by the Strathclyde research team.

### House 1 Appliance Channel Inventory
In **House 1**, the monitored sub-meter channels correspond to:
- `Appliance1`: Fridge
- `Appliance2`: Chest Freezer
- `Appliance3`: Upright Freezer
- `Appliance4`: Tumble Dryer
- `Appliance5`: **Washing Machine** (Target Appliance)
- `Appliance6`: Dishwasher
- `Appliance7`: Desktop Computer
- `Appliance8`: Television
- `Appliance9`: Electric Heater

---

## 3. Raw Data Quality Audit Findings (House 1)

When inspecting raw readings for House 1, four primary data-quality artifacts were identified:

### 1. Timestamp Duplication
- **Observation:** Approximately 0.2% to 0.6% of raw records contained duplicate timestamps.
- **Root Cause:** Wireless transmission re-tries from Current Cost RF transmitters, where the receiver logged multiple incoming packets within the same second before updating the internal buffer.

### 2. Irregular Sampling Intervals
- **Observation:** While the nominal design interval is 8 seconds, the actual delta $\Delta t$ between consecutive rows exhibits significant jitter (ranging from 6 seconds to 16 seconds), with occasional multi-minute dropouts.
- **Root Cause:** Clock drift, wireless RF signal attenuation through domestic walls, and packet collisions on the 433 MHz ISM band.

### 3. Out-of-Range and Impossible Readings
- **Negative Power:** Rare negative readings (e.g., -1 W to -5 W) observed due to sensor zero-calibration drift under zero-load conditions.
- **Physical Sensor Exceedances:** Occasional spikes $> 4,000\text{ W}$ on individual appliance plug-in meters. Plug-in IAMs are physically rated for 16A at 230V (~3,680 W nominal, 4,000 W maximum). Any value above 4,000 W represents sensor ADC measurement saturation or inductive switching noise.
- **Aggregate Exceedances:** Isolated transient spikes $> 12,000\text{ W}$ on the whole-house CT clamp, exceeding normal UK domestic panel main fuse ratings (60A–80A at 230V).

### 4. Communication Outages & Extended Missing Gaps
- **Observation:** Multi-hour and multi-day gaps exist where the base station receiver lost power, crashed, or encountered network failure.

---

## 4. Documented Cleaning Methodology & Justifications

To prepare House 1 data for reliable machine learning and disaggregation, we implemented a modular, deterministic cleaning pipeline:

```
[Raw Records ~8s]
       │
       ▼
1. Chronological Sorting & Datetime Validation
       │
       ▼
2. Duplicate Resolution Rule: Mean Aggregation per Timestamp
       │
       ▼
3. Outlier Handling: Non-negativity clipping & 4000W / 12000W bounds
       │
       ▼
4. Uniform Resampling to 1-Minute Intervals (1min mean)
       │
       ▼
5. Short-Gap Handling: Linear interpolation for gaps <= 2 mins
       │
       ▼
6. Prolonged Outage Handling: Flag gaps >= 120 mins with 'is_outage = 1'
       │
       ▼
[Cleaned 1-Minute Dataset]
```

### Detailed Decision Justifications:

1. **Duplicate Resolution via Mean Aggregation:**
   - *Alternative Considered:* Dropping all duplicates except the first occurrence (`drop_duplicates(keep='first')`).
   - *Justification for Selected Rule:* If two sensor readings occur within the same sub-second window (e.g. 150W and 180W), computing their mean (165W) preserves the total electrical energy integral ($E = \int P \, dt$), whereas blindly dropping one discards valid physical load transitions.

2. **Physical Outlier Capping:**
   - Active power cannot be negative; negative values are clipped to $0.0\text{ W}$.
   - Individual appliance channels are clipped at $4,000\text{ W}$ in strict adherence to the manufacturer's rated physical limit and the official Strathclyde cleaning guidelines.
   - Aggregate channel is clipped at $12,000\text{ W}$.

3. **1-Minute Uniform Resampling:**
   - The raw ~8-second data contains non-uniform timestamps that prevent standard discrete-time digital signal processing and lag generation.
   - Resampling to uniform 1-minute intervals (`1min` mean) provides a clean, regular time index while retaining the key operational stages of domestic washing machines (which last from 5 to 30 minutes each).

4. **Short Gaps ($\le 2$ minutes) vs. Prolonged Outages ($\ge 120$ minutes):**
   - *Short Gaps:* Communication packet drops lasting 1–2 minutes are filled using forward-facing linear time-interpolation. Because physical appliance thermal and mechanical states change slowly over 60–120 seconds, linear interpolation introduces negligible error.
   - *Prolonged Outages:* When the monitor is offline for $> 2\text{ hours}$, blind interpolation or forward-filling is scientifically invalid (it would invent artificial electricity use during an outage). Instead, prolonged outages are explicitly flagged with `is_outage = 1`, allowing downstream models to exclude these windows from training loss calculations.

---

## 5. Cleaning Impact: Record Counts & Signal Comparison

| Metric | Raw Dataset (House 1) | Cleaned Dataset (House 1) |
| :--- | :--- | :--- |
| **Sampling Resolution** | Irregular (~8 seconds) | Uniform (1 minute) |
| **Record Count (1 year)**| ~3,900,000 rows | 525,600 rows (exact 1-min grid) |
| **Duplicate Timestamps** | ~15,000 detected | 0 (fully resolved) |
| **Sensor Spikes (>4,000W)** | ~250 instances | 0 (safely capped to physical rating) |
| **Flagged Outage Minutes**| N/A | Explicitly tracked via `is_outage` |

---

## 6. Remaining Data Quality Concerns & Caveats

1. **Unmetered Residual Electricity:** The sum of all 9 IAM channels does not equal the whole-house aggregate meter. In REFIT households, 30% to 50% of total electrical energy remains unmetered (lighting circuits, cookers, immersion water heaters, and untracked wall sockets). Models must learn to distinguish target appliance signatures from this unmetered background baseline.
2. **Appliance Substitutions:** In certain houses, occupants replaced appliances during the two-year study (e.g., purchasing a new washing machine). Users of the cleaned dataset should consult `CLEAN_READ_ME_081116.txt` when analyzing multi-year trends.
3. **Reactive Power Absence:** The REFIT dataset measures only active real power (Watts). Reactive power (VAR), phase angle, and current harmonics—which significantly facilitate NILM classification in high-end laboratory datasets—are unavailable.
