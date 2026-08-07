# A synthetic study of baseline CRT-HD scores and lifecycle maintenance costs in a fleet of load-bearing widgets

*Author's note: this paper is entirely invented. It was written only to
exercise the *meltiro* test suite. No real study, component, operator, depot,
or result is described. Every name, number, and finding below is fictional.
Any resemblance to an actual publication or dataset is coincidental. The
findings are nonetheless presented in ordinary technical prose so that a
data-extraction pipeline has genuine, self-consistent content to extract.*

Halloran M, Okonkwo B, Fairweather S, Nakamura T, Delacroix P.

Journal of Synthetic Widget Reliability Engineering (invented), 2021.

DOI: 10.0000/demo.0001

## Abstract

Background. Load-bearing widgets account for a large share of fleet
maintenance expenditure. Routinely recorded inspector-rated durability gauges
such as the Composite Rig Test (Heavy Duty), abbreviated CRT-HD, might help
identify units at risk of high resource use.

Aims. To estimate associations between baseline CRT-HD total scores and
subsequent maintenance costs, durability index, and unplanned removal in a
fleet of load-bearing widgets.

Methods. Retrospective cohort of 512 load-bearing widgets with a rated working
load of 18 to 65 kN, held on the structural bracket register of the Northmoor
Regional Rail Authority depot network. Baseline CRT-HD was rated by depot
inspectors as part of routine grading. Units were followed for 12 months.

Results. Each one-point increase in baseline CRT-HD total score was associated
with an additional GBP 412 in annual overhaul cost (95% CI 250 to 574,
p < 0.001, adjusted). Higher CRT-HD scores correlated with a lower DI-4
durability index (Pearson r = -0.38, 95% CI -0.47 to -0.28, p < 0.001). A
baseline CRT-HD total of 15 or more was associated with increased odds of
unplanned removal within 12 months (odds ratio 1.87, 95% CI 1.29 to 2.71,
p = 0.001).

Conclusions. In this synthetic fleet, baseline CRT-HD scores were associated
with maintenance costs, durability index, and unplanned removal. Routinely
collected CRT-HD data may have value for maintenance planning.

## Introduction

Load-bearing widgets, including structural brackets and related coupling
assemblies, are associated with high overhaul and inspection expenditure.
Gauges that flag units likely to incur high costs could support maintenance
planning. The Composite Rig Test (Heavy Duty) is a 12-item inspector-rated
gauge recorded routinely across the depot network, and its predictive value
for lifecycle outcomes is of interest.

The primary aim of this study was to estimate the association between
baseline CRT-HD total score and annual overhaul cost. Secondary aims were to
examine associations between baseline CRT-HD and the DI-4 durability index,
inspection visit frequency, unplanned removal, and inspector-confirmed
fatigue cracking.

## Methods

Design. Retrospective cohort study using linked routine inspection and
maintenance-accounting records.

Setting and population. Load-bearing widgets with a rated working load of 18
to 65 kN in service on the structural bracket register of the Northmoor
Regional Rail Authority, a regional depot network. Units whose primary duty
was decorative cladding or temporary propping were excluded.

Case identification. Eligible units were identified from the depot asset
register. All units on the register with an index installation between
1 January 2016 and 31 December 2018 were included. Records were accessed
retrospectively; no unit was removed from service for this study.

Data source. Data were drawn from routinely collected asset-management
records, linked at unit level to the grading dataset (which includes CRT-HD
ratings), workshop activity records, and DI-4 durability index readings
gathered during a routine fleet condition survey. Costs were derived from
activity data using the 2019/20 depot tariff and published component unit
costs. These data were collected for operational and accounting purposes, not
primarily for research.

Exposure. The exposure was the baseline CRT-HD total score, rated by the depot
inspector at the index installation survey as part of routine grading. CRT-HD
has 12 items each scored 0 to 4 (total 0 to 48); higher scores indicate
greater degradation.

Outcomes. The primary outcome was annual overhaul cost. Secondary outcomes
were the DI-4 durability index (scored with the heavy-duty tariff), scheduled
inspection visits per year, unplanned removal within 12 months, and
inspector-confirmed fatigue cracking within 12 months. Fatigue cracking was
defined as an inspector-documented crack requiring escalation to workshop
repair, judged on inspection and independently of any gauge threshold.

Follow-up. Units were followed for 12 months from the index survey.
Observation ran to 31 December 2019.

Sample size. 512 units met the inclusion criteria.

Statistical analysis. Multiple linear regression was used for annual cost,
Pearson correlation for the DI-4 durability index, negative binomial
regression for inspection visit counts, logistic regression for unplanned
removal, and Cox proportional hazards regression for time to fatigue
cracking. Except where noted, regression models were adjusted for unit age,
duty class, manufacturer, and baseline load severity.

Data governance. The study was approved by the (fictional) Eastfield Asset
Data Committee (reference 21/SY/0000) and received depot engineering
approval. As anonymised routine records were used, individual operator
consent was not required.

## Results

Of 512 units (mean age 41 months, 58% heavy duty class), all had a baseline
CRT-HD rating. The mean baseline CRT-HD total score was 14.2 (SD 6.1).

Table 1 reports the primary and secondary associations estimated by
regression.

| Relationship | Gauge score | Outcome | Method | Effect estimate | Uncertainty |
|---|---|---|---|---|---|
| R1 | Baseline CRT-HD total | Annual overhaul cost (GBP) | Multiple linear regression | beta = 412 GBP per point | 95% CI 250 to 574; p < 0.001 |
| R2 | Baseline CRT-HD total | DI-4 durability index | Pearson correlation | r = -0.38 | 95% CI -0.47 to -0.28; p < 0.001 |
| R3 | Baseline CRT-HD total | Inspection visits per year | Negative binomial regression | IRR = 1.06 per point | 95% CI 1.03 to 1.09; p < 0.001 |

The cost and visit models (R1 and R3) were adjusted for unit age, duty class,
manufacturer, and baseline severity; the durability index correlation (R2)
was unadjusted.

Two further associations were estimated but not tabulated. Units with a
baseline CRT-HD total of 15 or more had higher odds of unplanned removal
within 12 months than those scoring below 15 (adjusted odds ratio 1.87, 95%
CI 1.29 to 2.71, p = 0.001; logistic regression adjusted for unit age, duty
class, and manufacturer). In a Cox proportional hazards model, each one-point
increase in baseline CRT-HD total score was associated with a higher rate of
inspector-confirmed fatigue cracking within 12 months (hazard ratio 1.09 per
point, 95% CI 1.04 to 1.14, p = 0.002, adjusted for unit age, duty class,
manufacturer, and baseline severity).

## Discussion

In this synthetic fleet, higher baseline CRT-HD scores were associated with
greater overhaul costs, a lower durability index, more frequent inspection
visits, and higher risks of unplanned removal and fatigue cracking. The
consistency of direction across outcomes is what would be expected if CRT-HD
captured overall structural need. As the data are routine and drawn from a
single fictional depot network, residual confounding and incomplete recording
would be important limitations in any real analysis.

## Conclusion

Baseline CRT-HD scores, collected routinely across the depot network, were
associated with a range of maintenance-cost, durability, and reliability
outcomes in this invented widget fleet. The dataset is fictional and exists
only as deterministic input for the *meltiro* test suite.
