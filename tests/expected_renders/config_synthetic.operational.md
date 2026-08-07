# Extraction template (operational)

## Study-level extraction

### Identity

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Study label**<br>`study_label` | Text | No | optional | Short label used in tables | Usually first author surname plus year. It is constructed rather than quoted, so evidence is optional. |
| **Title**<br>`title` | Text | No | required | Full paper title |  |
| **Authors**<br>`authors` | Text (multiple) | No | required | Author list, one per entry |  |
| **Year**<br>`year` | Year | No | required | Publication year |  |
| **Journal**<br>`journal` | Text | No | required | Journal or source name |  |
| **DOI**<br>`doi` | Text | No | required | Digital Object Identifier |  |

### Publication

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Abstract**<br>`abstract` | Text | No | required | The paper's abstract | Reproduce the abstract as printed. It stands in as the study's short summary when the paper bundle's manifest carries none. |
| **Publication type**<br>`publication_type` | Journal article; Technical report; Standards body publication; Other (specify) | No | required | Kind of document this is |  |

### Design and setting

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Design**<br>`design` | Bench test; Field trial; Retrospective service records | No | required | Design as described by authors |  |
| **Widget class**<br>`widget_class` | Text | No | required | Class of widget studied |  |
| **Sample size**<br>`sample_size` | Number | No | required | Widgets in the analysed sample |  |
| **Test start**<br>`test_start` | Date | No | required | When data collection began |  |

### Gauge Exposure

_Section extraction instruction._ Record what the study measured, not what it found. A gauge belongs here whenever the paper reports collecting it, even if no relationship is estimated from it.

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Gauges collected**<br>`gauges_collected` | Names from the Gauge Reference List | No | required | Gauges recorded in this study | One exact name per gauge recorded, as a list. |
| **Mean gauge score**<br>`mean_gauge_score` | Number | No | required | Mean gauge score reported |  |

## Record-level extraction

_Entity._ `relationship` (plural: relationships)

_Description._ a reported relationship between a durability gauge score and a lifecycle outcome

_Record extraction instruction._ Create one entry per distinct gauge-outcome statistical relationship reported. Walk each results table row by row and emit a separate entry for every outcome variable named. Do NOT consolidate related outcomes into a single combined entry.

### Gauge

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Gauge**<br>`gauge` | Name from the Gauge Reference List | Yes | required | Which durability gauge is being assessed in this relationship | Use exact names from the Gauge Reference List. |
| **Gauge score format**<br>`gauge_score_format` | Text | No | required | How the gauge score is operationalised |  |

### Outcome

_Section extraction instruction._ Record an outcome only where the paper reports a direct statistical estimate tying a gauge score to it.

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Outcome variable**<br>`outcome_variable` | Text | Yes | required | The specific outcome measure |  |
| **Outcome category**<br>`outcome_category` | Cost or resource use; Service life; Failure state | Yes | optional | Broad outcome category |  |
| **Index tariff**<br>`index_tariff` | Text | No | required | Durability index tariff used | Only for Service life outcomes; leave null otherwise. |
| **Cost source**<br>`cost_source` | Text | No | required | Where the unit costs came from |  |
| **Failure state definition**<br>`failure_state_definition` | Text | No | required | How the failure state is defined |  |

### Statistical Relationship

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Statistical method**<br>`statistical_method` | Text | No | required | Analysis that produced the estimate |  |
| **Effect type**<br>`effect_type` | Text | No | required | Kind of estimate reported |  |
| **Effect size**<br>`effect_size` | Text | No | required | Point estimate as reported |  |
| **Direction**<br>`direction` | Positive; Negative; Null; Mixed | No | optional | Direction of the association |  |
| **Estimate basis**<br>`estimate_basis` | Primary analysis; Subgroup analysis; Sensitivity analysis; Other (specify) | No | optional | Analysis this estimate comes from |  |
| **Adjusted**<br>`adjusted` | Yes/No | No | required | Whether the estimate is adjusted |  |

## Study-level quality appraisal

### Reporting

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Qa reporting**<br>`qa_reporting` | Compliant; Non-compliant; Not reported | No | optional | Compliance with a reporting checklist | In this field's notes, name the checklist used. |

## Record-level quality appraisal

### Sample and Selection

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Rqa sample adequate**<br>`rqa_sample_adequate` | Adequate; Marginal; Inadequate; Unclear | No | optional | Was the sample adequate here |  |

### Method Appraisal

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Rqa pre specified**<br>`rqa_pre_specified` | Pre-specified; Exploratory; Unclear | No | optional | Pre-specified or exploratory analysis |  |

## Initial check

### Initial Check

_Section extraction instruction._ Recorded before extraction begins. The extractor scans the supplied inputs, flags any obvious problem, and commits to how many relationships it expects to find.

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Text readable**<br>`text_readable` | Yes/No | Yes |  | Is the supplied text readable |  |
| **Figure tables included**<br>`figure_tables_included` | Yes/No | Yes |  | Was every exhibit supplied |  |
| **Missing exhibits**<br>`missing_exhibits` | Text (multiple) | No |  | Exhibits missing from the bundle |  |
| **Expected relationships**<br>`expected_relationships` | Text | Yes |  | How many relationships are expected |  |

## Quality check

### Quality Check

_Section extraction instruction._ Recorded at mark_complete. The extractor reflects on what actually happened, including any divergence from `expected_relationships`.

| Field | Type / values | Required | Evidence | Description | Extraction instruction |
|:---|:---|:---|:---|:---|:---|
| **Deviation from expectations**<br>`deviation_from_expectations` | Text | Yes |  | How many were extracted, and why |  |
| **General notes**<br>`general_notes` | Text | No |  | Any other observation worth recording |  |
