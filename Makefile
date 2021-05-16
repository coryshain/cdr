cognition:
        python3 -m cdr.bin.train cognition_ini/synth_noise_e0.ini
        python3 -m cdr.bin.train cognition_ini/synth_noise_e1.ini
        python3 -m cdr.bin.train cognition_ini/synth_noise_e10.ini
        python3 -m cdr.bin.train cognition_ini/synth_noise_e100.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_fixed1.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_fixed5.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_rnd1.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_rnd5.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_async1.ini
        python3 -m cdr.bin.train cognition_ini/synth_time_async5.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.00.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.25.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.50.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.75.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.90.ini
        python3 -m cdr.bin.train cognition_ini/synth_multicollinearity_r0.95.ini
        python3 -m cdr.bin.train cognition_ini/synth_misspecification_e.ini
        python3 -m cdr.bin.train cognition_ini/synth_misspecification_n.ini
        python3 -m cdr.bin.train cognition_ini/synth_misspecification_g.ini
        python3 -m cdr.bin.train cognition_ini/natstor_raw.ini
        python3 -m cdr.bin.predict cognition_ini/natstor_raw.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/natstor_raw.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/natstor_raw.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/natstor_raw.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/natstor_raw.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/natstor_raw.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/natstor_log.ini
        python3 -m cdr.bin.predict cognition_ini/natstor_log.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/natstor_log.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/natstor_log.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/natstor_log.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/natstor_log.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/natstor_log.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_raw_sp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_sp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_sp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_sp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_raw_sp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_raw_sp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_raw_sp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_log_sp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_log_sp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_log_sp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_log_sp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_log_sp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_log_sp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_log_sp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_raw_fp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_fp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_fp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_fp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_raw_fp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_raw_fp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_raw_fp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_log_fp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_log_fp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_log_fp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_log_fp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_log_fp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_log_fp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_log_fp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_raw_gp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_gp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_gp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_raw_gp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_raw_gp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_raw_gp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_raw_gp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/dundee_log_gp.ini
        python3 -m cdr.bin.predict cognition_ini/dundee_log_gp.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/dundee_log_gp.ini -m CDR_G_bbvi -p test
        python3 -m cdr.bin.predict cognition_ini/dundee_log_gp.ini -m "(LM|GAM).*" -p test
        python3 -m cdr.bin.convolve cognition_ini/dundee_log_gp.ini -m "CDR_G_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/dundee_log_gp.ini -m "CDR_G_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/dundee_log_gp.ini -m "CDR_G_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/fmri.ini
        python3 -m cdr.bin.predict cognition_ini/fmri.ini -p train dev
        python3 -m cdr.bin.predict cognition_ini/fmri.ini -m CDR_HRF5_bbvi -p test
        python3 -m cdr.bin.convolve cognition_ini/fmri.ini -m "CDR_HRF5_bbvi.*" -p train dev-test
        python3 -m cdr.bin.lmer cognition_ini/fmri.ini -m "CDR_HRF5_bbvi.*" -p train dev-test -u
        python3 -m cdr.bin.lrt cognition_ini/fmri.ini -m "CDR_HRF5_bbvi.*" -a -p train dev-test
        python3 -m cdr.bin.train cognition_ini/fmri_convolved.ini
        python3 -m cdr.bin.predict cognition_ini/fmri_convolved.ini -p train dev test
        python3 -m cdr.bin.train cognition_ini/fmri_interpolated.ini
        python3 -m cdr.bin.predict cognition_ini/fmri_interpolated.ini -p train dev test
        python3 -m cdr.bin.train cognition_ini/fmri_averaged.ini
        python3 -m cdr.bin.predict cognition_ini/fmri_averaged.ini -p train dev test
        python3 -m cdr.bin.train cognition_ini/fmri_lanczos.ini
        python3 -m cdr.bin.predict cognition_ini/fmri_lanczos.ini -p train dev test
        python3 -m cdr.bin.bakeoff results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/CDR_G_bbvi/losses_mse_test.txt -b results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/LMnoS/losses_mse_test.txt -n CDR_v_LMnoS -o results/bakeoff
        python3 -m cdr.bin.bakeoff results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/CDR_G_bbvi/losses_mse_test.txt -b results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/LMEfullS/losses_mse_test.txt -n CDR_v_LMEfullS -o results/bakeoff
        python3 -m cdr.bin.bakeoff results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/CDR_G_bbvi/losses_mse_test.txt -b results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/GAMnoS/losses_mse_test.txt -n CDR_v_GAMnoS -o results/bakeoff
        python3 -m cdr.bin.bakeoff results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/CDR_G_bbvi/losses_mse_test.txt -b results/{natstor_raw,natstor_log,dundee_raw_sp,dundee_log_sp}/GAMfullS/losses_mse_test.txt -n CDR_v_GAMfullS -o results/bakeoff
	python3 -m cdr.bin.bakeoff results/fmri/CDR_HRF5_bbvi/losses_mse_test.txt -b results/fmri_convolved/LME/losses_mse_test.txt -n CDR_v_canonicalHRF -o results/bakeoff
	python3 -m cdr.bin.bakeoff results/fmri/CDR_HRF5_bbvi/losses_mse_test.txt -b results/fmri_interpolated/LME/losses_mse_test.txt -n CDR_v_interpolated -o results/bakeoff
	python3 -m cdr.bin.bakeoff results/fmri/CDR_HRF5_bbvi/losses_mse_test.txt -b results/fmri_averaged/LME/losses_mse_test.txt -n CDR_v_averaged -o results/bakeoff
	python3 -m cdr.bin.bakeoff results/fmri/CDR_HRF5_bbvi/losses_mse_test.txt -b results/fmri_lanczos/LME/losses_mse_test.txt -n CDR_v_lanczos -o results/bakeoff
	python3 -m cdr.bin.pt cognition_ini/natstor*.ini -a -p dev-test -M loglik
	python3 -m cdr.bin.pt cognition_ini/dundee*.ini -a -p dev-test -M loglik
	python3 -m cdr.bin.pt cognition_ini/fmri*.ini -a -p dev-test -M loglik
	python3 -m cdr.bin.pt cognition_ini/natstor*.ini -a -p dev-test -M loglik -t
	python3 -m cdr.bin.pt cognition_ini/dundee*.ini -a -p dev-test -M loglik -t
	python3 -m cdr.bin.pt cognition_ini/fmri*.ini -a -p dev-test -M loglik -t

