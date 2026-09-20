# Boundary-phase precheck

Passed. The train/live contract is 32 frames at 0.27s and 0.53s, with exactly 100 visible glosses and three internal phases. Incomplete O5S5 rows can create KNOWN targets only; UNKNOWN and TRANSITION come only from fully annotated ASLLRP rows. All 1,166 raw archives passed schema and hash checks. A full dry run produced 38,043 training and 9,380 validation context windows with no incomplete-row negative targets. The real MPS tiny-fit reduced loss from 1.0120 to 0.0143. Citizen test was not accessed.
