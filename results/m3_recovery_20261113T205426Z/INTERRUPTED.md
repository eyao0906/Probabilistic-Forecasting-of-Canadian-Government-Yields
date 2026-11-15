# Interrupted targeted run

This directory contains a partial M3 recovery run, not a completed comparison.
It was stopped after three fit records (2023-02-24, variants A–C) because the
comparison CSV writer did not yet accept the union of historical and new
diagnostic columns. Those fit records and the training-only regularization
selection outputs are retained as execution evidence; no summary or full
forecast comparison was produced from this run. The corrected implementation
was rerun into a separate timestamped directory. Do not treat these partial
records as the final experiment or overwrite them.
