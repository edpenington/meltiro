# Can SPLINE-CD build a corrosion index score? Adapting a durability-score method to rig-wide gauge data

*Author's note: this paper is entirely invented. It was written only to exercise the meltiro test suite. No real study, package, depot, sample, or result is described, and any resemblance to an actual publication or dataset is coincidental. The methods are nonetheless set out in ordinary technical prose, with the typography and notation of a converted journal article, so that a data-extraction pipeline has genuine, self-consistent content to work on.*

Ferré J, Bäcklund M, Okonkwo B, Halloran M, Delacroix P.

Journal of Synthetic Widget Reliability Engineering (invented), 2025.

DOI: 10.0000/syn.0003

## Abstract

### Objective

SPLINE-CD builds a composite durability score from the summary statistics of a network-wide gauge association study, and it does so by leaning on a covariance map that tells it which gauges move together. Corrosion has no such map. We asked whether the package can be pointed at rig-wide gauge data anyway, with the covariance prior replaced by something a depot can actually supply: clusters of gauges that corrode together, blocks of bays that sit next to one another, or a plain window sliding along the running line. Each variant was scored on held-out corrosion readings from withdrawn and in-service widgets (*N* = 1,308). Nothing in the design required raw readings to leave the depot that recorded them, a condition of access.

### Results

Two of the three substitutes worked as well as the method they were meant to improve on and no better, explaining about 4.2% of the variance in scrappage. The co-corroding clusters were the exception, and they failed for a mundane reason: too few gauges ended up in a cluster with anything else. What is more telling is that random clusters, carrying no structural information at all, matched the bay-based prior almost exactly. On this evidence the gain comes from the sampler rather than from the prior, and choosing between priors is not where effort is best spent. None of this is an argument against the package, only against the assumption that a better prior is where the next improvement lies.

## Introduction

Corrosion is now measured across a whole fleet at once. A rig-wide gauge association study (RGAS) puts a reading from every gauge against an outcome and reports which associations survive, in the same way that a network-wide gauge association study (NGAS) does for design features [1, 2]. What comes out is a table of effect sizes, and the usual thing to do with such a table is to collapse it into one number per unit: a corrosion index score (CIS), the counterpart of the composite durability score (CDS) built from NGAS output. Whether that number is worth anything depends entirely on what went into it, which is the subject of the rest of this paper.

Scores of this kind already track scrappage well enough to be interesting [3]. Our own attempt, built by pruning correlated gauges and thresholding what was left, accounted for 4.0% of the variance in scrappage [4]. Pruning and thresholding is, however, the crude end of the literature. Better methods exist on the CDS side, all of them running from summary statistics so that depots can pool without pooling raw data, and most of them borrowing from Bayesian regression, regularisation, or Markov chain Monte Carlo [5–7]. SPLINE-CD is the best known of these. Its sampler needs a prior that says which gauges carry redundant information, and on the CDS side that prior is a gauge covariance (GC) map read off a reference fleet [5, 8]. The appeal is practical: a depot that cannot share readings can still share effect sizes, and the sampler asks for nothing else.

No such map exists for corrosion, and none is likely to: two gauges correlate because they sit in the same wet corner, not because of anything fixed about the design. So the question is whether a substitute prior will do. Three were tried, in decreasing order of how much structure they claim to encode:

- Co-corroding regions (CCRs), which group gauges that both sit close together and read alike across a reference set [10, 11].
- Topologically adjacent bays (TABs), which group gauges by the depot bay they fall in and claim nothing about their readings [11].
- A window of fixed length sliding along the running line, which claims nothing at all beyond proximity.

## Main text

### Data

Training summary statistics came from the pooled scrappage analysis reported previously (*N* = 2,415) [4]. Held-out readings came from the Northmoor depot register, covering withdrawn and in-service widgets from the same period (*N* = 1,308) [12]. Readings in the held-out set had already been residualised against unit age, duty class, estimated coating loss, batch, and the leading principal components of both manufacturer and gauge position, so what is modelled here is the corrosion left over once the obvious sources of variation are removed [4]. Units withdrawn during the period were kept with their last complete set of readings, since dropping them would have removed most of the outcome.

### Methods

Analyses ran under CoGauge 0.1.0 and bigrig 1.12.2, the latter supplying both rigstat and SPLINE-CD [10, 13]. Versions are pinned because the clustering behaviour of CoGauge changed between minor releases.

#### SPLINE-CD

The sampler is Gibbs and the framing is Bayesian: posterior effect sizes come from the summary statistics and the prior together [5]. What matters for our purposes is that the prior can be handed in as a fixed set of blocks rather than derived on the fly, and that inside each block the package wants a sparse pairwise correlation matrix over the gauges. Blocks are passed as an index rather than a matrix, so memory is set by the largest block, not by the gauge count.

#### CoGauge

CoGauge is what produced the blocks. It walks the gauge list, joins a gauge to a cluster when the gauge is close enough and its readings agree well enough, and stops a cluster growing when the run of unclustered sites between two members gets too long [10]. Three settings govern that behaviour: *corlo*, the correlation floor; *maxgaugedst*, the furthest two gauges may be and still join; and *corlodst*, the longest gap of intervening sites tolerated inside one cluster. None of the three settings has an obvious default, and the package documentation is candid about that. That gap is acknowledged upstream.

### Implementation

Every model used the auto variant of the sampler, which fits *h*^*2*^ and *p* from the data and so needs no validation split [5]. For each prior in turn the gauges were clustered by the process described below, the pairwise correlations were computed inside each block, and the resulting matrix was handed to the sampler. Gauges absent from either the training or the held-out set were dropped before any of this. The auto variant was chosen over the grid on cost, a grid search over this many block sets being beyond our compute budget.

### Co-corroding regions

Settings were chosen to cluster as much as possible without becoming meaningless: *corlo* = 0.2, *maxgaugedst* = 100,000 mm, and *corlodst* = 800 mm. A second matrix was then built from the same run with every singleton cluster removed, on the argument that a cluster of one contributes nothing the sampler can use. That second matrix returned a negative *h*^*2*^, which the sampler will not accept, so the value was pinned at 10^− 5^ for the run to proceed and the result is reported with that caveat attached. A looser correlation floor was tried first and produced clusters so large that the sparse matrix stopped being sparse. Longer correlation runs were not attempted.

### Sliding window approach

Setting *corlo* = 10^− 10^ turns CoGauge into a pure proximity clusterer: the correlation test can never fail, so gauges group by position along the bay and nothing else. Discarding correlation at this stage costs less than it appears to, because the sampler computes pairwise correlation inside each block regardless [5]. Six window lengths were run, at 5 metres (m), 10 m, 20 m, 100 m, 500 m and 1 kilometre (km), with *maxgaugedst* and *corlodst* both pinned to the window length under test. The window runs along the line rather than across it, so gauges on opposite faces of one bracket cluster together only by accident.

### Topologically adjacent bays (TAB)

The bay scaffold is the one published by Ravensworth & Quayle [11], downloaded on March 15, 2024 and mapped onto depot layout revision 19 [13]. It gives consensus start and end positions for every bay, pooled across seven layouts, which is what we wanted: the held-out units are stabled across mixed stock and a layout-specific scaffold would have had to assume a composition we do not know. Gauge positions were taken from the Ashfield 450k and Brentmoor wide-span manifests shipped with CoGauge and each gauge was assigned to the bay containing it, giving 2640 blocks. Bay boundaries were taken as published; where a gauge fell exactly on a boundary it was assigned to the lower-numbered bay.

### Random clusters

If the bay scaffold contributes anything, a model built on clusters of the same shape but no structure should do worse. Ten such null sets were generated. Across the 2,640 bay blocks the gauge counts ran min = 1, Q~1~ = 38, median = 74, mean = 106, Q~3~ = 129 and max = 2,410, which is close enough to log-normal to imitate: cluster sizes (*P*) were drawn as *P* ~ LogNorm(mean = log(74), SD = 0.8) and set counts (*C*) as *C* ~ Norm(mean = 2640, SD = 132), so each null set holds about as many clusters of about the same sizes as the real scaffold. Gauges were then assigned to those clusters at random. Two things about it are worth flagging. The null sets match the scaffold on the marginal distribution of cluster sizes but not on which gauges share a cluster, which is the comparison we want; and because the sizes are drawn rather than fixed, the ten sets differ from one another about as much as any differs from the scaffold itself.

### Applying and evaluating scores

A score is a matrix product, **S** = **M**^**T**^**β**, with **M**~**(gauges x units)**~ the held-out readings and **β** the posterior effect sizes the sampler returns. Scores were regressed on the recorded scrappage outcome by logistic regression, exactly as in the earlier work so that the numbers are comparable [4], and each model is reported by its Nagelbrink R^2^ [9] alongside the p-value and AIC from the same fit. Intervals on the R^2^ are not reported: the resampling needed to compute them honestly was outside our compute budget.

## Results & discussion

Table 1 sets the four families side by side: the two CCR matrices, the six sliding windows, the bay scaffold, and the ten null sets, each against the pruning-and-thresholding baseline. The headline is how little separates them. Every model except the CCR matrix that kept its singletons lands within a few thousandths of the baseline on explained variance and within a couple of AIC points of it. The ordering of the models is stable across the ten null sets, which is the only thing about the ordering that is.

**Table 1 Fit of each prior against the baseline**

| Model | *p*-value | Nagelbrink *R*^2^ | AIC | Gauges clustered* |
| --- | --- | --- | --- | --- |
| P + T (Halloran et al. 2024) | 3.49 × 10^− 5^ | 0.0401 | 1712.4 | 14% |
| SPLINE-CD, CCR blocks | 0.518 | 0.000512 | 1744.9 | 21% |
| SPLINE-CD, CCR blocks, no singletons | 6.72 × 10^− 6^ | 0.0308 | 1721.6 | 21% |
| SPLINE-CD, CCR sliding window blocks** | [2.11 to 2.44] × 10^− 7^ | 0.0417 to 0.0420 | 1710.1 to 1710.6 | 71–88% |
| SPLINE-CD, TAB blocks | 1.94 × 10^− 7^ | 0.0423 | 1709.8 | 99% |
| SPLINE-CD, random cluster blocks | [2.35 to 3.06] × 10^− 7^ | 0.0409 to 0.0415 | 1710.4 to 1711.3 | 99% |

* Share of gauges landing in a cluster with at least one other gauge. For *SPLINE-CD*,* CCR blocks*,* no singletons* the figure is also the share of gauges the model saw at all, the rest having been discarded with their clusters

** Window-by-window figures are in Appendix II

CCR: co-corroding region, TAB: topologically adjacent bay

Why the singleton-bearing CCR matrix failed is visible in the matrix itself (Appendix I): most of the diagonal is bare, and a sampler given a block structure that covers a fifth of the gauges has almost no correlation to work with. Dropping the singletons fixed the coverage without fixing the conditioning, and the negative variance component is the symptom of that; the run completed only because the value was pinned, and the estimate it produced is not one we would defend. The honest summary of the CCR arm is that CoGauge, run at settings loose enough to be usable, does not cluster corrosion gauges densely enough for this sampler. A matrix that is mostly diagonal is not a prior in any useful sense; it is an assertion that the gauges are independent.

The bay scaffold came top, but by so little that the ranking is not worth much on its own. What settles the question is the null model: ten sets of random clusters, matched only on how many clusters there are and how big they are, scored within a thousandth of the scaffold. Whatever the bays encode about which gauges belong together, the sampler is not using it. The gain over pruning and thresholding is real but it comes from the regularisation, not from the prior. That is a negative result about the scaffold, not about the bays, which may well matter for something else. The practical effect is small either way.

The sliding windows say the same thing from another direction. A 5 m window leaves nearly a third of gauges in clusters of one and still scores level with the scaffold, and lengthening the window past 10 m changes the distribution of cluster sizes far more than it changes the fit (Appendix II). If a depot has correlations available at any stage of its pipeline, position alone is a serviceable prior. The two shortest windows differ from each other more than either differs from the scaffold.

There is a structural reason to expect this. On the CDS side, a gauge array is surveyed at a few thousand positions and the covariance map is then used to interpolate the readings in between, taking the analysis from 400,000 positions to something nearer 11,000,000 [7]; pruning afterwards is how the redundancy that interpolation introduced gets removed again. RGAS has neither step. Every gauge in a rig-wide study was physically read, so there is no interpolated redundancy to prune, and the structural reference that pruning would need matters correspondingly less. Neither observation is new on the CDS side; what is new is that it survives the move to corrosion.

What we can say is that SPLINE-CD ports to corrosion data without much trouble and buys a small, consistent improvement over pruning and thresholding, which reproduces the earlier estimate that around 4.0% of scrappage variance is attributable to measurable corrosion [4]. What we cannot say is that any of the three priors earned its keep. If the useful step is limiting how many gauges enter the score, then regularisation does that directly and a prior is a detour [5].

Both packages here rest on pairwise correlation, and CoGauge sits inside the baseline as well as the new models [4], so the comparison is less independent than it looks. Methods that do not assume linear pairwise structure, random forests among them, would be the natural next thing to try.

### Limitations

- One dataset, one outcome, one network; nothing here speaks to a fleet of a different age or duty.
- The blocks were derived from the held-out set rather than from an independent one, which leaks a little information into the models and would inflate any of the estimates above.
- The auto sampler tunes its own hyperparameters, and Ferré et al. report it landing close to the grid search [5]; we did not run the grid search to check that on corrosion data.
- Training and held-out sets of 2,415 and 1,308 are small for this kind of model, and the differences we are trying to resolve are of the order of a thousandth of an R^2^.
- Coating renewal history was not available for the held-out units and could confound the corrosion-scrappage association directly.

## Appendix I

*Fig. 1 Two block correlation matrices, drawn to the same scale. Left: the intended structure, with shaded blocks along the diagonal and everything off-block empty. Right: what the CCR run actually produced once singletons were included, where most of the diagonal is bare because a cluster of one contributes no off-diagonal entries at all*

## Appendix II

**Table 2 Cluster-size distribution and fit, window by window and for the bay scaffold**

| Window size | Gauges clustered* | Min | Q1 | Q2 | Mean | Q3 | Max | *p*-value | Nagel-brink R^2^ | AIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | 71% | 1 | 1 | 2 | 2.14 | 3 | 84 | 2.44 × 10^− 7^ | 0.0417 | 1710.6 |
| 10m | 76% | 1 | 1 | 2 | 2.38 | 3 | 107 | 2.29 × 10^− 7^ | 0.0418 | 1710.4 |
| 20m | 80% | 1 | 1 | 2 | 2.62 | 3 | 107 | 2.21 × 10^− 7^ | 0.0419 | 1710.2 |
| 100m | 84% | 1 | 1 | 2 | 2.94 | 4 | 107 | 2.15 × 10^− 7^ | 0.0420 | 1710.1 |
| 500m | 88% | 1 | 1 | 2 | 3.01 | 4 | 107 | 2.11 × 10^− 7^ | 0.0419 | 1710.1 |
| 1km | 88% | 1 | 1 | 2 | 3.01 | 4 | 107 | 2.13 ×10^− 7^ | 0.0419 | 1710.1 |
| **TAB clusters** | **99.98%** | **1** | **38** | **74** | **106.2** | **129** | **2410** | **1.94 × 10** ^**− 7**^ | **0.0423** | **1709.8** |

^*^Share of gauges landing in a cluster with at least one other gauge

TAB: topologically adjacent bay

Coverage stops improving above 500 m and the size distribution is essentially settled by 10 m. None of this bears on the choice between priors. Neither the coverage nor the size distribution tracks the fit: the widest window and the narrowest differ by a factor of three in mean cluster size and by less than a thousandth in the corrosion index scores’ explained variance

## References

1. Ashgrove TP, Bellworthy RN, Corrigan DL, Draycott HM, Elmslie JW, Fenwick AR, et al. Running a network-wide gauge association study: quality control and analysis in practice. J Depot Methods. 2018;27:e118. https://doi.org/10.0000/jdm.2018.0027

2. Garnock TE, Hesketh MB, Illingworth CS, Jessamy AL, Kirkbride TO, Lampitt DR, et al. Rig-wide association studies: what is settled, what is not, and what to report. Corros Rep. 2021;13:66. https://doi.org/10.0000/cr.2021.0013

3. Merrivale KP, Naysmith EJ, Oldbury HT, Pentreath SV, Quiller AB, Rushworth PN, et al. Does corrosion mapping illuminate the mechanical pathways to premature scrappage? Depot Sci. 2022;52:410-425. https://doi.org/10.0000/ds.2022.0052

4. Halloran M, Okonkwo B, Fairweather S, Nakamura T, Delacroix P. A corrosion index score for scrappage risk in a synthetic bracket fleet. Synth Struct Notes. 2024;29:77. https://doi.org/10.0000/ssn.2024.0029

5. Ferré J, Bäcklund M, Sowerby ID. SPLINE-CD: a faster sampler for composite durability scores. Rig Data J. 2021;36:1140-1148. https://doi.org/10.0000/rdj.2021.0036

6. Tredinnick LF, Umbers GK, Vellacott DM, Wintersgill JA, Yardley CB, Zeal MR, et al. Bayesian multiple regression on summary statistics improves composite prediction. Synth Eng Commun. 2019;10:512. https://doi.org/10.0000/sec.2019.0010

7. Zeal MR, Umbers GK, Ashgrove TP, Bellworthy RN, Corrigan DL, Draycott HM, et al. Ten composite scoring methods for withdrawal outcomes, compared across depot cohorts. Struct Cond Rev. 2021;90:188-197. https://doi.org/10.0000/scr.2021.0090

8. Elmslie JW. Gauge covariance as a record of service history and a guide to maintenance. Struct Eng Rev. 2008;9:220-228. https://doi.org/10.0000/ser.2008.0009

9. Nagelbrink NJD. A general coefficient of determination for depot models. J Synth Stat. 1991;78:203-210. https://doi.org/10.0000/jss.1991.0078

10. Fenwick AR, Garnock TE, Hesketh MB, Jessamy AL. CoGauge: finding co-corroding regions in gauge array data. Rig Data J. 2020;36:733-741. https://doi.org/10.0000/rdj.2020.0036

11. Quiller AB, Pentreath SV. Depot bays in three dimensions: building a consensus scaffold. Depot Architecture Rev. 2015;16:88-97. https://doi.org/10.0000/dar.2015.0016

12. Nyström A, Kirkbride TO, Lampitt DR, Merrivale KP, Naysmith EJ, Oldbury HT, et al. Divergent corrosion after early-service overload in withdrawn bracket stock. Applied Depot Eng. 2024;14:9. https://doi.org/10.0000/ade.2024.0014

13. Ferré J, Rushworth PN, Sowerby ID, Tredinnick LF. Handling network-scale gauge data with two R packages: rigstat and bigrig. Rig Data J. 2018;34:602-609. https://doi.org/10.0000/rdj.2018.0034
