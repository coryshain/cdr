default:
	# TRAINING
	# Main analyses
	python -m cdr.bin.train fMRI_ns_LANG.ini
	python -m cdr.bin.train fMRI_ns_MD.ini
	python -m cdr.bin.train fMRI_ns_combined.ini
	# Follow-up analyses splitting on documents
	python -m cdr.bin.train fMRI_ns_LANG_splitdoc.ini
	python -m cdr.bin.train fMRI_ns_MD_splitdoc.ini
	python -m cdr.bin.train fMRI_ns_combined_splitdoc.ini
	# Follow-up analysis using spatial WM localizer for MD
	python -m cdr.bin.train fMRI_ns_MDSPWM.ini
	# EXPLORATORY EVALUATION
	python -m cdr.bin.predict fMRI_ns_LANG.ini -p train
	python -m cdr.bin.predict fMRI_ns_MD.ini -p train
	python -m cdr.bin.predict fMRI_ns_LANG_splitdoc.ini -p train
	python -m cdr.bin.predict fMRI_ns_MD_splitdoc.ini -p train
	python -m cdr.bin.predict fMRI_ns_MDSPWM.ini -p train
	# CRITICAL (GENERALIZATION) EVALUATION
	python -m cdr.bin.predict fMRI_ns_LANG.ini -m CDR_full -p dev-test
	python -m cdr.bin.predict fMRI_ns_MD.ini -m CDR_full -p dev-test
	python -m cdr.bin.predict fMRI_ns_combined.ini -m CDR_full -p dev-test
	python -m cdr.bin.predict fMRI_ns_LANG_splitdoc.ini -m CDR_full -p dev-test
	python -m cdr.bin.predict fMRI_ns_combined_splitdoc.ini -m CDR_full -p dev-test
	# SIGNIF TESTING
	python -m cdr.bin.ct fMRI_ns_LANG.ini -m CDR_full -a -T -p dev-test
	python -m cdr.bin.ct fMRI_ns_combined.ini -m CDR_full -a -T -p dev-test
	python -m cdr.bin.ct fMRI_ns_LANG_splitdoc.ini -m CDR_full -a -T -p dev-test
	python -m cdr.bin.ct fMRI_ns_combined_splitdoc.ini -m CDR_full -a -T -p dev-test
	
