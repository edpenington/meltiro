The checker views one field at a time, with a small amount of context sufficient to inform its judgement. For each field it receives:

- a description of the field that was extracted;
- a short identity label: a summary of the paper, and, for a record-level field, a one-line record label carrying the record id and any context fields;
- the evidence the extractor provided: one or more verbatim quotes, each shown first as a quoted block and then again inside the paper's own surrounding text with the quoted span marked, and/or references to cropped exhibits, which are attached to the message as images;
- the value the extractor claims;
- the extractor's note on this field, when it wrote one. A note is commentary rather than evidence, and nothing in it has been verified against the paper, so the checker weighs it as an account of the extractor's reasoning and not as a source. A note is often absent, and its absence says nothing either way.

The question is a single one: does the provided evidence, read in its context in the paper, support the extracted value?

The surrounding text is there to settle what the evidence means. A quote can be unreadable on its own: a table cell says `1.34` without saying whether that is a hazard ratio or a mean difference, and a number carries no unit until its column header supplies one. So a quote taken from a table in the paper text arrives with the lines above it, back to the top of that table, and that expansion is never trimmed to fit the context budget, however far above the cell the header row sits. The table's caption comes with it when the paper sets that caption on its own line just above the table, which is usual but not guaranteed; a caption printed below the table, folded into a running paragraph, or held off by other text reaches the checker only if it happens to fall inside the ordinary window, so an absent caption means the caption could not be resolved, not that the table has none. Where a table arrives as a cropped image instead, the image is the whole table, headings included.

That context is the paper's own words and nothing else. It is not the extractor's argument and must not be read as one: the extractor's reasoning reaches the checker through the note on the field, never through the paper text around a quote.

The checker's view is still deliberately narrow. It sees a neighbourhood of each quote rather than the paper as a whole, and of the extraction it sees only what the identity label carries. It does not take responsibility for the wider quality of the extraction process; it is interested only in whether the supplied evidence justifies the supplied value, given the description of the field.

The extractor's actions are also constrained, and the checker is aware of it. For categorical fields the per-field message lists the field's allowed or typical values: a strict-list field requires a match to one of the options, whereas an open-list field treats them as typical values and may legitimately carry free text when none applies. Some fields are validated against a reference list and must carry an exact name from it; spelling on those is already validator-guaranteed.

Each check is a single, self-contained judgement. The checker is asked about one field, once, and answers on the material in front of it. It holds no memory of any earlier check and is never told what became of a verdict it gave. A field that arrives after the extractor revised it arrives as a fresh question with no history attached.
