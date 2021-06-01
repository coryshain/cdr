acl21:
        python3 -m cdr.bin.train acl_ini/natstor.ini
        python3 -m cdr.bin.predict acl_ini/natstor.ini -p train dev test
        python3 -m cdr.bin.train acl_ini/dundee.ini
        python3 -m cdr.bin.predict acl_ini/dundee.ini -p train dev test
        python3 -m cdr.bin.train acl_ini/natfmri.ini
        python3 -m cdr.bin.predict acl_ini/natfmri.ini -m CDRNN_FF CDRNN_RNN -p train dev test
        python3 -m cdr.bin.train acl_ini/synth_noise_e10.ini
        python3 -m cdr.bin.train acl_ini/synth_time_async1.ini
        python3 -m cdr.bin.train acl_ini/synth_multicollinearity_r0.75.ini

