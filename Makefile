acl21:
	python3 -m cdr.bin.train ini/dundee_raw_sp.ini
	python3 -m cdr.bin.predict ini/dundee_raw_sp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_raw_sp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/dundee_log_sp.ini
	python3 -m cdr.bin.predict ini/dundee_log_sp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_log_sp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/dundee_raw_fp.ini
	python3 -m cdr.bin.predict ini/dundee_raw_fp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_raw_fp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/dundee_log_fp.ini
	python3 -m cdr.bin.predict ini/dundee_log_fp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_log_fp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/dundee_raw_gp.ini
	python3 -m cdr.bin.predict ini/dundee_raw_gp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_raw_gp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/dundee_log_gp.ini
	python3 -m cdr.bin.predict ini/dundee_log_gp.ini -p dev
	python3 -m cdr.bin.predict ini/dundee_log_gp.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/natstor_raw.ini
	python3 -m cdr.bin.predict ini/natstor_raw.ini -p dev
	python3 -m cdr.bin.predict ini/natstor_raw.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/natstor_log.ini
	python3 -m cdr.bin.predict ini/natstor_log.ini -p dev
	python3 -m cdr.bin.predict ini/natstor_log.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/fmri.ini
	python3 -m cdr.bin.predict ini/fmri.ini -p dev
	python3 -m cdr.bin.predict ini/fmri.ini -m CDR_main -p test
	python3 -m cdr.bin.train ini/fmri_main.ini
	python3 -m cdr.bin.predict ini/fmri_main.ini -p dev test
	python3 -m cdr.bin.train ini/synth_noise_e0.ini
	python3 -m cdr.bin.train ini/synth_noise_e1.ini
	python3 -m cdr.bin.train ini/synth_noise_e10.ini
	python3 -m cdr.bin.train ini/synth_noise_e100.ini
	python3 -m cdr.bin.train ini/synth_time_fixed1.ini
	python3 -m cdr.bin.train ini/synth_time_fixed1.ini
	python3 -m cdr.bin.train ini/synth_time_rnd5.ini
	python3 -m cdr.bin.train ini/synth_time_rnd5.ini
	python3 -m cdr.bin.train ini/synth_time_async1.ini
	python3 -m cdr.bin.train ini/synth_time_async5.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.00.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.25.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.50.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.75.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.90.ini
	python3 -m cdr.bin.train ini/synth_multicollinearity_r0.95.ini
	python3 -m cdr.bin.train ini/synth_misspecification_e.ini
	python3 -m cdr.bin.train ini/synth_misspecification_g.ini
	python3 -m cdr.bin.train ini/synth_misspecification_n.ini

