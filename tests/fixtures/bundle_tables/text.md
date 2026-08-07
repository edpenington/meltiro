# Does a high flexural durability grade protect load-bearing widgets from cracking? A one-year depot cohort study

*Author's note: this paper is entirely invented. It was written only to exercise the meltiro test suite. No real study, component, operator, depot, or result is described, and any resemblance to an actual publication or dataset is coincidental. The findings are nonetheless set out in ordinary technical prose, with the tables and the typography of a converted journal article, so that a data-extraction pipeline has genuine, self-consistent content to work on.*

Halloran M, Okonkwo B, Fairweather S, Nakamura T, Delacroix P.

Journal of Synthetic Widget Reliability Engineering (invented), 2024.

DOI: 10.0000/syn.0002

## Abstract

### Objectives

We asked whether a widget graded highly for flexural durability at one inspection round is less likely to be found cracked at the next round.

### Methods

Two inspection rounds twelve months apart supplied the data, the first in February 2020 and the second in February 2021. Eligibility was restricted to units rated over 18 kN. Any widget missing from either round, or already cracked when the first round opened, was set aside. Ten rig-sheet items make up the depot durability grade, and all ten were scored at the first round. Flexural failure at follow-up was determined from the inspector's written finding of a through-crack or a fractured load path. Odds ratios (OR) came from multivariate logistic regression carrying unit age, duty class, manufacturer, baseline load severity, and any workshop repair recorded before the first round as fixed covariates.

### Results

Of the 588 widgets carried through both rounds, n = 47 (8.0%) were found cracked at the second. Each additional grade point lowered the odds of cracking once covariates were allowed for (aOR = 0.79 [95% confidence interval: 0.70–0.89]). Splitting the fleet at the cut-off read off these data reproduced the pattern: units graded 6 or above against those graded below (< 6) gave (aOR = 0.42 [0.21–0.84]).

### Conclusions

The grade carries usable warning of cracking a year ahead. Confirmation against a withdrawal standard set independently of the rig sheet is the obvious next step. A threshold fit for depot scheduling is proposed.

## Introduction

Cracking at a load path is what takes most load-bearing widgets out of service before their planned life, and the cost falls twice: on the workshop that renews the unit and on the timetable that loses it [1]. Once a first path has opened, a second usually follows [2], bearings run hot [3], and the unit is scrapped early rather than repaired [4]. Degradation short of a reportable crack is far commoner on a depot register than cracking itself [5], and it is not harmless: it drives extra inspection visits [6], keeps units off the roster [7], and raises whole-life cost [8, 9]. If anything is to be done cheaply, it has to be done at that stage. What a depot needs is a handful of rig-sheet items that separate the units worth watching from the rest [10, 11], resting on properties an engineer can act on rather than on history [12]. Nothing in current practice does that well [13]. Grading also has to be quick enough that a busy depot will do it, which rules out anything needing a jig or a second visit.

Flexural durability was proposed as a way of scoring what a sound unit has rather than what a failing one lacks [13]. The construct gathers several things a rig can see: no open defect on any load path, measurable residual stiffness, and enough redundancy, tolerance and shock margin that a single overload does not propagate [14]. None of those alone is the grade; the point of the construct is that they interact, and that the attending inspector's overall reading of the unit carries information no single item does [15]. Ten items were selected to represent it, and the resulting flexural durability grade (FDG) has been described in full elsewhere [16, 17]. Read at a single round, the grade falls as crack counts rise [18] and as rig-measured deflection grows [19], and it separates a unit merely degrading from one that has already failed [19]. Work on repeated rounds hints that a well-graded unit deteriorates more slowly [14, 20], but nobody has yet followed a graded fleet forward to see whether the grade anticipates cracking that has not happened yet. Two features matter for what follows. It is scored on what the inspector can see without disturbing the unit, so a grade costs a walk round and a torch rather than a possession; and it is bounded at both ends, so a unit sitting at the ceiling cannot be told apart from a better one, which is a real limitation on any fleet fresh out of overhaul. Neither has been examined against a future outcome.

That is the question here. We follow a graded fleet for one year, ask whether the grade at the first round predicts cracking at the second, and look for a threshold a depot could act on. No unit was withdrawn, re-rostered or repaired on account of the study.

## Methods

### Study design

Widgets were graded once, left in ordinary service for twelve months, and graded again. The first round ran across the network in February 2020; the second ran in February 2021. The Eastfield Asset Data Committee of the Northmoor Regional Rail Authority, a fictional body, reviewed and approved the work under reference 20/SY/0000. No grade was passed back to the scheduling office, so a grade could not trigger the repair it might predict.

### Units

Candidates came from the structural bracket register of the Northmoor Regional Rail Authority. The register carries roughly 2.3 million inspection events, and slots for the first round went out in the order entries came up for routine attention until the target count was met; 1140 units were graded. To be eligible an entry had to sit on the structural bracket register and to be rated at 18 kN or above. The first round had a purpose of its own, comparing grades between units carrying a history of major workshop repair but no current defect and units carrying neither, so slots were split evenly between the two strata, roughly 50% against 50%. Every eligible entry rated over 18 kN and stabled inside the network was offered a slot. Grades were entered by the attending inspector on the standard rig sheet, and a sheet returned blank counted as no grade rather than as a zero. All 1140 graded units were called back twelve months later. Depots received 100 tokens of rig time for each round they completed, which covered the time the sheet took. Two categories of entry were excluded before slots were offered: units whose primary duty was decorative cladding or temporary propping, since neither carries a rated load path, and units already booked for withdrawal inside the following twelve months.

Two further conditions applied to the longitudinal sample reported below: a unit had to be graded at both rounds, and it had to be free of cracking at the first.

### Measurement gauges

#### Flexural durability grade (FDG)

Ten items make up the depot version of the rating sheet [13]. An inspector marks each item absent (0) or present (1) and the marks are summed, so a unit scores between 0 and 10 and a higher score is the better one. Reliability and validity of this version were established in earlier work and are not re-examined here [19]. Items are marked independently and no item is weighted above another.

#### Flexural cracking

Two questions on the rig sheet carry the outcome, asked of the interval since the previous round (“Did the unit show a visible crack along any load path during the round?” “Was the unit removed from service for unplanned workshop repair between the rounds?”), a pairing known at the depots as the Whitlow items. Each takes No (0) or Yes (1), and a Yes to either counts the unit as cracked. Neither question exhausts what the Depot Manual of Structural Defects would call a withdrawable defect, and neither is meant to; between them they flag a unit worth a closer look [11]. Both questions ask about the unit rather than the inspector's opinion of it: an item asking whether a crack was likely would be answered from the grade.

#### Register variables

Five fields were pulled from the register for each unit: duty class (frontline or second line), age at the index survey, mounting type (single-bolt or twin-bolt), manufacturer (Ashfield Forge, Brentmoor Castings, Calder Works or other), and whether a major workshop repair had been recorded before the first round. That last field combines two register items, crack repair and load-path renewal, read across the whole service life to date. None of the five fields was missing for more than 2% of units.

### Statistical analysis

Mean grades were compared across the register characteristics one at a time, using univariate analysis of variance.

Detection characteristic (DC) curve analysis supplied sensitivity, specificity, the two predictive values, and the area under the curve (AUC), and located the cut-off. Conventional bands were used to describe the area: above 0.9 excellent, 0.8 to 0.9 good, 0.7 to 0.8 fair, 0.6 to 0.7 poor, and 0.5 to 0.6 no better than a coin [21]. The curve was computed on complete cases and no interpolation was used.

Odds of cracking at the second round were regressed on the grade at the first by multivariate logistic regression, with unit age, duty class, mounting type, manufacturer, and prior workshop repair as covariates. Three parameterisations of the exposure are reported: the grade as a continuous score, the grade split at the cut-off, and the grade cut into quantiles. Units missing a covariate were dropped from the adjusted models and kept in the crude ones.

Two-sided p < 0.05 was taken as significant throughout. Analyses ran under release 28.0 of the depot analysis toolkit (Northmoor Analysis Office, Eastfield).

## Results

Both rounds were completed for 588 units. Table 1 sets out the units’ characteristics; mean age at the index survey was 57.4 months (standard deviation; 15.2), and 40.1% carried a workshop repair from before the first round. Table 2 breaks the mean grade down by register characteristic. Grades ran high in older stock (96 months or more), in twin-bolt mountings, and in units with no repair behind them. Grades spread across the whole range, with no floor or ceiling effect.

**Table 1 Units’ Characteristics at Baseline (N = 588)**

|  | N (%) | Mean (SD) [min—max] |
| --- | --- | --- |
| Age at index survey |  | 57.4 (15.2) [18–96] |
| Under 24 months | 46 (7.8) |  |
| 24–47 months | 118 (20.1) |  |
| 48–71 months | 139 (23.6) |  |
| 72–95 months | 163 (27.7) |  |
| 96 months or more | 122 (20.7) |  |
| Duty class |  |  |
| Frontline | 341 (58.0) |  |
| Second line | 247 (42.0) |  |
| Mounting type |  |  |
| Single-bolt | 194 (33.0) |  |
| Twin-bolt | 394 (67.0) |  |
| Manufacturer |  |  |
| Ashfield Forge^(a)^ | 231 (39.3) |  |
| Brentmoor Castings | 162 (27.6) |  |
| Calder Works or over | 195 (33.2) |  |
| Workshop repair before the first round^(b)^ |  |  |
| Yes | 236 (40.1) |  |
| No | 352 (59.9) |  |

*SD* standard deviation

(a) One sheet gave the maker as “other” (n = 1) and is counted under Ashfield Forge

(b) Prior workshop repair combines two register items, crack repair and load-path renewal, read over the whole service life to date; a Yes on either item puts the unit in the repaired group, and a blank on both items reads as no repair rather than as missing

**Table 2 Flexural Durability Grade at Baseline, by Register Characteristic and Group Difference (N = 588)**

|  | Mean score of the flexural durability grade at baseline [possible range 0–10] | Group difference |
| --- | --- | --- |
| Mean (SD) | p-value ^(a)^ |  |
| Age at index survey |  |  |
| Under 24 months | 5.9 (3.2) | < 0.001* |
| 24–47 months | 6.4 (2.9) |  |
| 48–71 months | 6.5 (2.7) |  |
| 72–95 months | 6.9 (2.6) |  |
| 96 months or more | 7.6 (2.4) |  |
| Duty class |  |  |
| Frontline | 6.5 (2.9) | 0.284 |
| Second line | 6.8 (2.8) |  |
| Mounting type |  |  |
| Single-bolt | 6.1 (3.1) | 0.004* |
| Twin-bolt | 7.0 (2.6) |  |
| Manufacturer |  |  |
| Ashfield Forge^(b)^ | 6.6 (2.8) | 0.712 |
| Brentmoor Castings | 6.7 (2.9) |  |
| Calder Works or over | 6.8 (2.7) |  |
| Workshop repair before the first round^(c)^ |  |  |
| Yes | 5.4 (3.0) | < 0.001* |
| No | 7.4 (2.5) |  |

*SD* standard deviation

^*^p < 0.05

(a) Group differences were tested one characteristic at a time by univariate analysis of variance

(b) One sheet gave the maker as “other” (n = 1) and is counted under Ashfield Forge

(c) Prior workshop repair combines two register items, crack repair and load-path renewal, read over the whole service life to date; a Yes on either item puts the unit in the repaired group, and a blank on both items reads as no repair rather than as missing

Cracking was recorded for n = 47 (8.0%) of the fleet at the second round. Figure 1 plots the detection characteristic curve. Area under it was 0.736 [95% confidence intervals: 0.661–0.803, p < 0.001], with the best operating point falling at a grade of 5.5. At that point sensitivity, specificity and the two predictive values stood at 71%, 58%, 13% and 96% (Table 3). Because the grade is an integer between 0 and 10, the working threshold was rounded up to 6. The curve, with its uncertainty band, is reproduced as Figure 1.

*Fig. 1 Detection characteristic curve of the flexural durability grade for predicting flexural cracking one year later*

**Table 3 Performance of the Flexural Durability Grade in Predicting Flexural Cracking at 1-year Follow-up (N = 588)**

| Flexural durability | Flexural cracking at follow-up round (1 year) |  |
| --- | --- | --- |
| Negative | Positive |  |
| Low (< 6), n (%) | 198 (86.1) | 32 (13.9) |
| High (6 or more), n (%) | 343 (95.8) | 15 (4.2) |
| Value (95% CI) |  |  |
| Sensitivity | 0.71 (0.60–0.82) | – |
| Specificity | 0.58 (0.49–0.67) | – |
| Positive predictive value | 0.13 (0.07–0.19) | – |
| Negative predictive value | 0.96 (0.92–1.00) | – |

*CI* confidence intervals

Table 4 carries the three regressions. Read as a continuous score, a high FDG score significantly predicted a lower rate of flexural cracking once the covariates were allowed for (aOR = 0.79 [95% confidence interval: 0.70–0.89]). Split at 6, the better-graded group repeated the finding (aOR = 0.42 [0.21–0.84]). Cut into quantiles, the top band (grades 9 and 10) sat furthest from the first band (grades 0 to 3), the crude and adjusted estimates differing little, at (OR = 0.29 [0.13–0.64]).

**Table 4 Odds ratio for Flexural Cracking at the Follow-up Round, Three Logistic Models (N = 588)**

|  | N | Crude | Adjusted^(a)^ |  |  |
| --- | --- | --- | --- | --- | --- |
| OR | 95% CI | OR | 95% CI |  |  |
| Model 1 (continuous) |  |  |  |  |  |
| Flexural durability grade | 588 | 0.74 | 0.66–0.83 | 0.79 | 0.70–0.89 |
| Model 2 (cut-off)^(b)^ |  |  |  |  |  |
| Low (score < 6) | 230 | 1.00 |  | 1.00 |  |
| High (6 or more) | 358 | 0.27 | 0.14–0.51 | 0.42 | 0.21–0.84 |
| Model 3 (quantile)^(c)^ |  |  |  |  |  |
| Quantile 1 (score 0–3) | 118 | 1.00 |  | 1.00 |  |
| Quantile 2 (4–5) | 141 | 0.41 | 0.20–0.84 | 0.44 | 0.21–0.92 |
| Quantile 3 (6–8) | 166 | 0.28 | 0.13–0.60 | 0.35 | 0.16–0.78 |
| Quantile 4 (9, 10) | 163 | 0.17 | 0.08–0.37 | 0.29 | 0.13–0.64 |

*OR* is the odds ratio and *CI* the confidence interval

(a) Covariates were unit age, duty class, mounting type, manufacturer, and prior workshop repair

(b) The grade was split at its median to form the two groups of Model 2

(c) The grade was cut at its quartiles to form the four bands of Model 3

## Discussion

A grade taken at one round carried real information about the next. Discrimination was fair rather than excellent, which is about what a ten-item sheet completed at the rig in a few minutes ought to deliver, and the brevity of the sheet is the reason it gets completed at all.

The stock that graded well was older, twin-bolt mounted, and free of repair history, and none of that is surprising. What the grade is built to capture is margin under changing duty, which is another word for redundancy and tolerance [14]. Older stock that survived its early culling has demonstrated margin rather than acquired it [4, 22], and generous mounting tolerance supplies it directly [16]. That units without a repair record grade higher was known before this study [19]. The reason to prefer such a composite to a service history is that a history cannot be changed and a tolerance can [12]: the grade points at something a depot is able to act on, which is the whole argument for grading at all. The alternative reading, that a well-graded unit is simply a lightly used one, cannot be dismissed here, although duty class enters every adjusted model and barely moves the estimates.

What the analysis will not support is a screening claim in the other direction. With the threshold at 6, the negative predictive value was 96%, so a well-graded unit is very unlikely to be found cracked within the year, and a depot can defer it with some confidence. The positive predictive value was 13%, so a poorly graded unit is usually not going to crack either, and treating a low grade as a prediction of failure would waste most of the work that it triggered. The fleet entered the study uncracked, so the spread of grades at the first round is a spread of margin and not of damage; it is the margin, not the grade itself, that any intervention has to move. Read that way the quantile model is the informative one, since it shows the odds falling steadily across the range rather than switching at a threshold, and a threshold is a scheduling convenience rather than a statement about the units. A depot should take two numbers away from this. The first is the negative predictive value, because that is the number that licenses deferring a unit, and it is high. The second is the rate of cracking in the fleet, because a low rate keeps the positive predictive value low however good the grade is, and no improvement to the sheet changes that arithmetic.

## Practical implications

Because the finding held whether or not a unit carried a repair history, the grade can be used where a history is unavailable or unreliable, which covers most transfers between operators. Anything that raises margin should raise the grade, and the candidates are ordinary workshop measures: re-cutting a closed-up tolerance, renewing a sacrificial shim before it is consumed, relieving pre-stress left by a rebuild [20]. Whether those measures move the outcome as well as the grade is not tested here, and a fleet given them ought to be graded again to find out. None of this follows from the analysis; it follows from what the grade is made of.

## Limitations

Five things bound what can be read from this analysis. First, no target sample size was fixed in advance; a post-hoc calculation using the observed AUC of 0.736, 80% power and a likelihood ratio of 9 puts the requirement at 231 units. Second, nothing here establishes reach beyond the Northmoor network, and depots with newer rigs and fuller record keeping were over-represented among those that completed both rounds. A depot that keeps its slot list current is also a depot that grades carefully, so the units in this analysis may be better documented than the register as a whole, which would push the association away from the null. Third, the outcome rests on the two Whitlow items alone, and those items are known to be generous: a sensitivity of 96% and a specificity of 57% against workshop confirmation put the positive predictive value near a third, so most of what they flag is not a withdrawable defect. Coating craze and paint shrinkage both read as a visible crack from a walking inspection and neither was excluded. Nor was severity recorded, so a hairline that closed under load counts alongside a path that had parted, and the cracking rate reported above is best read as an upper bound. A withdrawal standard set independently of the rig sheet would settle the point and none was available. Fourth, half the units at the first round were drawn deliberately from those carrying a major repair, so the repaired fraction here is higher than the register's, and the fleet is more heterogeneous than a random sample would have been; both push the apparent validity of the grade upward. Fifth, the covariate set was whatever the register happened to carry. Duty cycle, load spectrum and coating age are all plausible confounders and none of the three is recorded well enough to enter a model, so residual confounding cannot be ruled out and is the likeliest explanation for any part of the association that survives the other four caveats. None of the five is fatal, but together they set the ceiling.

## Conclusion

A grade taken at one round anticipates cracking at the next, and a threshold of 6 separates a group that can be safely deferred from one that cannot. A high FDG score significantly predicted a lower rate of flexural cracking after adjustment, and ten items completed at the rig were enough to do it. Whether the same threshold holds outside the Northmoor network is untested, and a fleet under a harsher duty cycle than this one would very likely need one of its own. The grade is no substitute for the workshop's judgement about a unit already flagged; it is a way of spending the inspection budget where the margin is thinnest.

## References

1. Ashgrove TP, et al. Load-path defects and incident bearing seizure across a bracket fleet. J Synth Struct Eng. 2020;24:118-127. https://doi.org/10.0000/jsse.2020.0024

2. Bellworthy RN, et al. Ten-year trajectories of surface degradation and the risk of fracture: a register-based study. Northmoor Depot Eng Rev. 2016;3:41-50. https://doi.org/10.0000/ndr.2016.0003

3. Corrigan DL, Draycott HM. Bearing temperature in the months after a first load-path failure. Struct Maint J. 2010;40:212-224. https://doi.org/10.0000/smj.2010.0040

4. Elmslie JW, et al. Early scrappage after a reportable crack: a meta-analysis of depot registers. J Asset Cond. 2004;83:9-18. https://doi.org/10.0000/jac.2004.0083

5. Fenwick AR, et al. How common is sub-threshold degradation? Evidence from the Northmoor Fleet Survey. J Asset Cond. 2020;265:77-85. https://doi.org/10.0000/jac.2020.0265

6. Garnock TE, et al. Inspection burden attributable to degradation short of withdrawal. Fleet Availability Rev. 2008;48:301-310. https://doi.org/10.0000/far.2008.0048

7. Hesketh MB, et al. Roster availability and minor degradation: a register-based study of nine depots. Acta Struct Synth. 2007;115:63-71. https://doi.org/10.0000/ass.2007.0115

8. Illingworth CS, Jessamy AL, Kirkbride TO. Whole-life cost of units carrying unreported degradation. J Asset Cond. 2004;79:144-153. https://doi.org/10.0000/jac.2004.0079

9. Lampitt DR, et al. Degradation markers as attributable risk factors for first-onset flexural failure. Arch Struct Eng. 1992;49:508-517. https://doi.org/10.0000/ase.1992.0049

10. Merrivale KP, et al. How well does a short rig sheet find a crack? Accuracy of the Whitlow items at first inspection. Eastfield J Struct Eng. 2018;212:14-22. https://doi.org/10.0000/ejse.2018.0212

11. Whitlow RG, et al. Two questions are enough: brief case-finding for structural degradation at the rig. J Gen Depot Pract. 1997;12:255-263. https://doi.org/10.0000/jgdp.1997.0012

12. Naysmith EJ. What can actually be changed? Modifiable targets for preventive intervention in ageing structural stock. Maint Futures. 2014;79:88-97. https://doi.org/10.0000/mf.2014.0079

13. Oldbury HT, Pentreath SV. Flexural durability as a construct, and why a depot might want one. J Struct Clinimetrics. 2016;85:3-12. https://doi.org/10.0000/jsc.2016.0085

14. Quiller AB, Oldbury HT. Mapping the engineering science of flexural durability. J Struct Clinimetrics. 2022;91:220-231. https://doi.org/10.0000/jsc.2022.0091

15. Oldbury HT, Rushworth PN, Sowerby ID. What the attending inspector adds: judgement in the depot inspection process. J Depot Eng. 2012;73:410-418. https://doi.org/10.0000/jde.2012.0073

16. Tredinnick LF, et al. Comparing scales for durability, redundancy and positive structural condition. J Asset Cond. 2021;294:31-38. https://doi.org/10.0000/jac.2021.0294

17. Tredinnick LF, et al. Item selection and scoring for the flexural durability grade. J Struct Clinimetrics. 2019;88:66-71. https://doi.org/10.0000/jsc.2019.0088

18. Umbers GK, et al. Crack counts against the regional version of the flexural durability grade. Struct Cond Pract. 2021. https://doi.org/10.0000/scp.2021.0001

19. Halloran M, Tredinnick LF, Okonkwo B. The depot version of the flexural durability grade: reliability, validity and sensitivity. Synth Struct Notes. 2021;21:145. https://doi.org/10.0000/ssn.2021.0021

20. Quiller AB, Oldbury HT. Flexural durability in maintenance practice: an emerging role for the grade. Struct Cond Rev. 2020;82:1904. https://doi.org/10.0000/scr.2020.0082

21. Vellacott DM. Reading a detection characteristic curve: first principles for depot engineers. Synth Meas Rev. 1978;6:1-9. https://doi.org/10.0000/smr.1978.0001

22. Wintersgill JA, et al. Structural redundancy in young and old bracket stock across four fleets. Int J Fleet Eng. 2012;27:150-158. https://doi.org/10.0000/ijfe.2012.0027
