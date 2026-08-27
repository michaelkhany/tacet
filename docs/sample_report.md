---
{
  "generated": "2026-08-26T22:33:16+00:00",
  "windows": 200,
  "parameters": [
    "tel_temp"
  ],
  "tip_count": 6,
  "tips": [
    {
      "code": "BLIND_INTERVAL",
      "severity": "critical",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 20,
      "start": "2026-03-01 10:00:00+00:00",
      "end": "2026-03-01 13:10:00+00:00",
      "evidence": {
        "contextual_missingness": 1.0,
        "observability_degradation": 1.0
      }
    },
    {
      "code": "STUCK_SENSOR",
      "severity": "warning",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 16,
      "start": "2026-03-01 22:20:00+00:00",
      "end": "2026-03-02 00:50:00+00:00",
      "evidence": {
        "flatline": 1.0
      }
    },
    {
      "code": "STUCK_SENSOR",
      "severity": "warning",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 11,
      "start": "2026-03-02 05:00:00+00:00",
      "end": "2026-03-02 06:40:00+00:00",
      "evidence": {
        "flatline": 1.0,
        "value_deviation": 1.0
      }
    },
    {
      "code": "TRUE_EXCURSION",
      "severity": "warning",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 4,
      "start": "2026-03-02 04:20:00+00:00",
      "end": "2026-03-02 04:50:00+00:00",
      "evidence": {
        "change_inconsistency": 1.0,
        "value_deviation": 1.0
      }
    },
    {
      "code": "DRIFT",
      "severity": "warning",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 4,
      "start": "2026-03-01 21:00:00+00:00",
      "end": "2026-03-01 21:30:00+00:00",
      "evidence": {
        "change_inconsistency": 0.658,
        "value_deviation": 0.177
      }
    },
    {
      "code": "EXPECTED_SILENCE",
      "severity": "info",
      "parameter": "tel_temp",
      "entity": "gpu-042",
      "span_windows": 15,
      "start": "2026-03-01 16:40:00+00:00",
      "end": "2026-03-01 19:00:00+00:00",
      "evidence": {
        "observability_degradation": 1.0
      }
    }
  ]
}
---

# Sample EII Cloud findings

Generated 2026-08-26T22:33:16+00:00 · 200 windows · 1 parameters · 6 tips

## Summary

**1** 🔴 critical · **4** 🟠 warning · **1** 🔵 info


**3 of 6 findings (50%) concern what could not be observed rather than what was measured.** Conventional threshold monitoring would not have raised them, because in every case the numbers that did arrive were unremarkable.

## Findings

### 1. Blind interval — the entity went quiet while it was supposed to be working

`BLIND_INTERVAL` · severity **critical** · 20 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-01 10:00:00+00:00 → 2026-03-01 13:10:00+00:00

**How to read this:** The cloud is dense here but the underlying metric line is absent. Samples stopped arriving while context said this entity had work in flight. This is not an idle period: it is a stretch of time you have no evidence about, and a conventional dashboard would render it as a flat or interpolated line that looks perfectly healthy.

**What to do:** Treat this window as unassessed rather than as normal. Check whether the exporter, the agent, or the node itself stopped responding, and whether the job scheduled here completed.

**Evidence at onset:** `contextual_missingness` = 1.0, `observability_degradation` = 1.0

### 2. Variance collapse — the signal arrives but has stopped changing

`STUCK_SENSOR` · severity **warning** · 16 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-01 22:20:00+00:00 → 2026-03-02 00:50:00+00:00

**How to read this:** The line is flat, in range, and on time. That combination passes every threshold check ever written, and it is what a frozen counter, a stuck sensor, or an exporter replaying a cached value looks like. Genuine telemetry from a live system is never this still.

**What to do:** Verify the value at the source. If it matches, the sensor is fine and the component is genuinely inactive; if the source has moved on, the collection path is stale and everything downstream of it is fiction.

**Evidence at onset:** `flatline` = 1.0

### 3. Variance collapse — the signal arrives but has stopped changing

`STUCK_SENSOR` · severity **warning** · 11 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-02 05:00:00+00:00 → 2026-03-02 06:40:00+00:00

**How to read this:** The line is flat, in range, and on time. That combination passes every threshold check ever written, and it is what a frozen counter, a stuck sensor, or an exporter replaying a cached value looks like. Genuine telemetry from a live system is never this still.

**What to do:** Verify the value at the source. If it matches, the sensor is fine and the component is genuinely inactive; if the source has moved on, the collection path is stale and everything downstream of it is fiction.

**Evidence at onset:** `flatline` = 1.0, `value_deviation` = 1.0

### 4. Clean excursion — genuinely out of envelope, and well observed

`TRUE_EXCURSION` · severity **warning** · 4 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-02 04:20:00+00:00 → 2026-03-02 04:50:00+00:00

**How to read this:** The value left its expected envelope while coverage stayed complete. This is the one signature on the chart you can take at face value: the measurement is trustworthy and the deviation is real.

**What to do:** Handle as an ordinary threshold event. Because observability is intact here, the magnitude and duration on the chart can be used directly.

**Evidence at onset:** `change_inconsistency` = 1.0, `value_deviation` = 1.0

### 5. Trajectory break — legal value, illegal rate of change

`DRIFT` · severity **warning** · 4 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-01 21:00:00+00:00 → 2026-03-01 21:30:00+00:00

**How to read this:** The value is still comfortably inside its envelope, so no threshold has been crossed, but it is moving at a rate that does not match its own history. Degradation shows up here long before it shows up as an excursion — this is the part of the chart where warning time is bought.

**What to do:** Extend the view backwards and check whether the slope is sustained. A sustained trajectory break with a known failure downstream is your lead-time signal; feed it to `tacet.analysis.lead_lag` to quantify the warning.

**Evidence at onset:** `change_inconsistency` = 0.658, `value_deviation` = 0.177

### 6. Silent, and expected to be — no cause for concern

`EXPECTED_SILENCE` · severity **info** · 15 windows · parameter `tel_temp` · entity `gpu-042` · 2026-03-01 16:40:00+00:00 → 2026-03-01 19:00:00+00:00

**How to read this:** No samples arrived, and context agrees none were due: nothing was scheduled on this entity. The gap in the chart is real but benign. This rule exists so that ordinary idle time does not compete for attention with genuine blind intervals, which look identical until context is consulted.

**What to do:** None. Shown so the gap is explained rather than merely absent.

**Evidence at onset:** `observability_degradation` = 1.0

## Parameters ranked by peak EII

| parameter | max_eii | mean_eii | plane |
| --- | --- | --- | --- |
| tel_temp | 1.0 | 0.5038 | telemetry |

## How to read an EII Cloud

The cloud scores six things. Three describe the machine, three describe your
ability to watch it:

| Component | What a high score means | Whose problem |
|---|---|---|
| `value_deviation` | reading left its expected envelope | the machine |
| `change_inconsistency` | legal value, abnormal rate of change | the machine |
| `flatline` | value arrives but has stopped moving | the sensor |
| `observability_degradation` | samples did not arrive | the pipeline |
| `contextual_missingness` | samples did not arrive *and were due* | the pipeline |
| `context_contradiction` | telemetry says idle, context says busy | either — establish order first |

A high total driven by the top two rows is an incident. A high total driven by
the bottom three is a **blind spot**: not evidence that something is wrong, but
evidence that you would not know if it were. The two are routinely confused, and
they call for opposite responses — the first one pages the hardware team, the
second one means your monitoring cannot currently answer the question.
