# Extraction template (publication)

## Study-level fields

| Section | Field | Description | Values |
|:---|:---|:---|:---|
| Identity | Study label | Short label used in tables | Text |
| Identity | Title | Full paper title | Text |
| Identity | Authors | Author list, one per entry | Text (multiple) |
| Identity | Year | Publication year | Year |
| Identity | Journal | Journal or source name | Text |
| Identity | DOI | Digital Object Identifier | Text |
| Publication | Abstract | The paper's abstract | Text |
| Publication | Publication type | Kind of document this is | Journal article; Technical report; Standards body publication; Other (specify) |
| Design and setting | Design | Design as described by authors | Bench test; Field trial; Retrospective service records |
| Design and setting | Widget class | Class of widget studied | Text |
| Design and setting | Sample size | Widgets in the analysed sample | Number |
| Design and setting | Test start | When data collection began | Date |
| Gauge Exposure | Gauges collected | Gauges recorded in this study | Names from the Gauge Reference List |
| Gauge Exposure | Mean gauge score | Mean gauge score reported | Number |

## Record-level fields

a reported relationship between a durability gauge score and a lifecycle outcome

| Section | Field | Description | Values |
|:---|:---|:---|:---|
| Gauge | Gauge | Which durability gauge is being assessed in this relationship | Name from the Gauge Reference List |
| Gauge | Gauge score format | How the gauge score is operationalised | Text |
| Outcome | Outcome variable | The specific outcome measure | Text |
| Outcome | Outcome category | Broad outcome category | Cost or resource use; Service life; Failure state |
| Outcome | Index tariff | Durability index tariff used | Text |
| Outcome | Cost source | Where the unit costs came from | Text |
| Outcome | Failure state definition | How the failure state is defined | Text |
| Statistical Relationship | Statistical method | Analysis that produced the estimate | Text |
| Statistical Relationship | Effect type | Kind of estimate reported | Text |
| Statistical Relationship | Effect size | Point estimate as reported | Text |
| Statistical Relationship | Direction | Direction of the association | Positive; Negative; Null; Mixed |
| Statistical Relationship | Estimate basis | Analysis this estimate comes from | Primary analysis; Subgroup analysis; Sensitivity analysis; Other (specify) |
| Statistical Relationship | Adjusted | Whether the estimate is adjusted | Yes/No |
