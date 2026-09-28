# notes

- looks like debug_script_d.py is the most contemporary
- looks like none of the scripts figured out accurately bounding columns
- h is much faster than g
- f_h_d skips a lot of distorted text that but misses a ton of material. Implementing one element from this in f to hopefully improve accuracy but false positives on extremely distorted text seems inevitable.
- f_f is the best so far but I can't figure out the occasional skipped section -- and the second sweep isn't a complete fix.