Most fields carry an `evidence` string alongside their `value`. Evidence is a single string mixing three elements in any order:

- **Verbatim quotes** wrapped in `<q>...</q>`. The text inside the tags must appear character-for-character in the paper text (after light whitespace, ligature, and smart-quote normalisation). Multiple quotes are written as multiple `<q>...</q>` blocks. ` ... ` or `[...]` inside a block elides intervening words; each side is verified against the paper and must appear there in the order written, so use separate `<q>` blocks for phrases from different passages; `[sic]` and `[emphasis added]` are accepted; inline reference markers like `[12]` may be omitted. Square brackets also mark text inserted into a quote: where a column header carries the percent sign and the cell reads `118 (35.6)`, quote it as `<q>118 (35.6[%])</q>`.
- **Image references** wrapped in `<img>label</img>`, where `label` is the exact filename stem of an attached cropped exhibit (`<img>table_02</img>`, not `<img>Table 2</img>`). Use one where the supporting evidence is a numeric cell or a visual element rather than a textual passage.
- **Brief interpretive prose** outside the tags. It is stored with the field, but it is not part of what the checker reads, so reasoning that has to travel with the value belongs in that field's `notes` instead.

When `value` is null, `evidence` may be null, or may briefly explain why the paper does not report the field. For most non-null fields the evidence string carries at least one `<q>` or `<img>` block; the tool schemas mark which fields permit pure prose or empty evidence. Derived values are fine when the quoted evidence justifies them: `test_start: Jan 2018` is supported by `<q>from January 2018 to December 2019</q>`.

Cropped exhibits attached to this study, each written as its label followed by the caption the paper gives it. Cite the label, never the caption.
{image_labels_list}
