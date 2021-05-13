npsy:
	python3 -m cdr.bin.train npsy_ini/lang.ini
	python3 -m cdr.bin.predict npsy_ini/lang.ini -p train dev-test
	python3 -m cdr.bin.train npsy_ini/md.ini
	python3 -m cdr.bin.predict npsy_ini/md.ini -p train dev-test
	python3 -m cdr.bin.train npsy_ini/combined.ini
	python3 -m cdr.bin.predict npsy_ini/combined.ini -p train dev-test
        python3 -m cdr.bin.pt npsy_ini/lang.ini -a -p dev-test 
        python3 -m cdr.bin.pt npsy_ini/md.ini -a -p dev-test 
        python3 -m cdr.bin.pt npsy_ini/combined.ini -a -p dev-test

