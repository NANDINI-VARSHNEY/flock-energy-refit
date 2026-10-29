# Executive Summary & Practical Recommendations for Electric Utilities

**Prepared for:** Utility Operations, Customer Experience, and Demand-Side Management (DSM) Teams  
**Project:** REFIT Appliance Disaggregation and Energy Reduction Pipeline  
**Target Appliance:** Domestic Washing Machines  

---

## 1. One-Page Utility-Facing Summary: Suitable Uses & Limitations

### Suitable & High-Value Uses of the Model:
1. **Targeted Customer Engagement & Behavioral Energy Audits:** Automatically estimate domestic washing machine cycle frequency, average wash duration, and energy per cycle from whole-home smart meter data without expensive sub-metering hardware.
2. **Automated Demand Response & Peak Shifting Verification:** Detect domestic laundry cycles during expensive grid peak hours (e.g., 17:00–20:00) and verify whether dynamic time-of-use (ToU) tariffs successfully encourage customers to shift laundry into off-peak or solar-surplus hours.
3. **Appliance Inefficiency & Anomaly Detection:** Flag households whose wash cycles consume $> 1.8\text{ kWh/cycle}$ or exhibit extended resistive heating times (indicating severe heating element limescale or thermostat malfunction) to offer appliance upgrade rebates.

### Unsuitable Uses (Operational Boundaries):
1. **Sub-Meter Legal Revenue Billing:** The model carries a mean absolute error of ~9 W and occasional false positives during complex cooking events. It must **not** be used for sub-metered legal billing or punitive penalties.
2. **Zero-Shot Transfer Across Heterogeneous Households:** Disaggregation models trained on a single dwelling experience elevated error (18%–28% energy error) when applied to unseen homes without baseline calibration.

---

## 2. Practical Energy-Saving Opportunity: "Wash Cold, Save Gold"

### The Customer Insight:
Our exploratory data analysis reveals that domestic washing machines consume **0.8 kWh to 1.3 kWh per cycle**, with **75% to 85% of that energy consumed solely by the internal resistive heating element** warming water from cold tap temperature (~15°C) to 40°C or 60°C. Mechanical tumbling and water pumping consume only ~0.15–0.25 kWh.

### Recommended Utility Intervention:
Deploy an automated mobile app / email alert program:
- **Notification Trigger:** When the disaggregation algorithm detects repeated high-temperature cycles (characterized by prolonged 2,200 W heating plateaus drawing $> 1.1\text{ kWh}$), the customer receives a personalized energy-saving tip.
- **The Behavioral Nudge:** *"Switching from 40°C/60°C to 30°C or Eco Cold-Wash reduces cycle electricity consumption by 50% (~0.5 kWh per load), saving approximately £35–£50 annually on laundry bills while preserving fabric longevity."*
- **Off-Peak Delay Start Recommendation:** Encourage customers to utilize the washing machine's built-in delay timer to run cycles during low-cost overnight or midday solar hours.

### How the Utility Measures Effectiveness (Rigorous Measurement & Verification):
A utility can rigorously validate this program using a **Randomized Control Trial (RCT)**:
1. **Cohort Selection:** Identify 5,000 households with active disaggregated washing machine usage.
2. **Experimental Design:** Randomly assign 2,500 households to the **Treatment Group** (receiving personalized cold-wash insights and off-peak prompts) and 2,500 households to the **Control Group** (receiving standard generic monthly statements).
3. **Key Performance Indicators (KPIs):**
   - **Average Energy per Cycle (kWh/cycle):** A successful cold-wash shift reduces mean energy per cycle from ~1.1 kWh to ~0.65 kWh.
   - **Peak-Hour Cycle Share (%):** Percentage of cycles initiated between 17:00 and 20:00 on weekdays.
4. **Statistical Verification:** Employ a Difference-in-Differences (DiD) econometric regression over a 6-month post-intervention period to isolate behavioral savings from weather or seasonal confounders.

---

## 3. Real-World NILM Limitations & Operational Risks

### 1. Overlapping High-Power Signatures
In residential dwellings, electric kettles (~2.5 kW), electric cooking hobs/ovens (~2.0–2.4 kW), and electric instantaneous showers (~7.5–9.5 kW) utilize resistive heating elements with power magnitudes that overlap heavily with washing machine water-heating phases. While temporal rolling features help filter out brief 3-minute kettle boils, multi-stage cooking events can cause false-positive detections.

### 2. Missing, Gapped, and Noisy Smart-Meter Data
Wireless communication dropouts and meter buffering gaps are common in smart meter rollouts. Blindly zero-filling or mean-imputing multi-hour communication dropouts distorts baseline usage estimates. The utility pipeline must employ explicit outage flagging (`is_outage`) and exclude missing intervals from verification calculations.

### 3. Differences Between Households
Households vary widely in occupant counts, laundry frequency (e.g. families with infants vs. single retirees), appliance age, and cycle preferences. A disaggregation model trained on one house cannot be assumed to generalize across an entire service territory without incorporating household demographic metadata or semi-supervised transfer learning.

### 4. Cross-Geographic Transfer Limitations: Applying UK Findings to India
Attempting to deploy a UK-trained washing machine disaggregation model in India will result in severe failure due to fundamental systemic and technological differences:
- **Appliance Archetype & Internal Heating:** In the UK, almost all domestic washing machines are **front-loading automatic units with built-in internal electric water heaters (2,000 W–2,400 W)**. In India, the vast majority of urban households use **top-loading semi-automatic or cold-fill fully-automatic pulsator machines** that have **no internal water heater** and draw only **200 W to 450 W** (drum motor only). The prominent 2,200 W heating signature that UK models rely upon simply does not exist in typical Indian households.
- **Water Heating Infrastructure:** If hot water is used in an Indian household, it is supplied externally from a standalone storage geyser (water heater, 2 kW–3 kW) rather than heated internally by the washing machine.
- **Power Quality & Voltage Volatility:** UK electricity supply is a stable 230V $\pm 10\%$ single-phase grid. Indian distribution networks frequently experience voltage fluctuations (180V–260V) and inverter/battery backup transitions that alter apparent appliance power signatures.

---

## 4. Impact of 15-Minute or 30-Minute Smart-Meter Data Granularity

In commercial AMI (Advanced Metering Infrastructure) rollouts, smart meters typically transmit readings every **15 or 30 minutes** to conserve cellular bandwidth and battery life. 

**Why 15/30-minute data makes disaggregation drastically harder:**
At 1-minute resolution, a washing machine cycle exhibits clear, identifiable state transitions: a sharp 20-minute 2,200 W heating plateau, distinct soak pauses, and a ramping 600 W spin cycle. At 15-minute or 30-minute intervals, these transient high-frequency features are averaged across the entire time bucket:
$$\bar{P}_{30\text{min}} = \frac{1}{30} \int_{0}^{30} P(t) \, dt$$
A sharp 2,200 W heating burst lasting 15 minutes followed by 15 minutes of 100 W tumbling is averaged into a flat ~1,150 W interval. A single 15-minute bucket blurs together the washing machine, a kettle boil, a microwave run, and the refrigerator. Step-change edges, duration boundaries, and motor pulse signatures are completely erased, degrading disaggregation from signature pattern recognition into coarse energy volume estimation. High-precision event detection becomes mathematically ill-posed without high-frequency data.
