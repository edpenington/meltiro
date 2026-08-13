The extraction record the reviewer sees was produced by an extractor using tools that validate against a known schema, to which the reviewer also has access. Every field in it passed validation before it was recorded, so a field whose value or evidence failed was rejected and never applied. Validation is a test of form, not of judgement: it confirms that a quote appears in the paper, but not whether that quote is the right passage, whether the value read off it is the right reading, or whether the record as a whole describes the study. That is what the reviewer is for.

Each field's `evidence` is a single string mixing `<q>...</q>` verbatim quotes from the paper text, `<img>label</img>` references to an attached cropped exhibit, and brief interpretive prose outside the tags. Within a `<q>` block, ` ... ` is a fragment separator: each side is verified against the paper and must appear there in the order written, so phrases from different passages belong in separate `<q>` blocks. When `value` is null the extractor may leave `evidence` null or use it to explain why the paper does not report the field.

The cropped exhibits attached to this study are as follows, each written as its label followed by the caption the paper gives it. An `<img>` citation carries the label, never the caption.

{image_labels_list}
