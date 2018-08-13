emnlp18:
    python -m dtsr.bin.train experiments_emnlp18\synth.ini
	python -m dtsr.bin.train experiments_emnlp18\natstor.ini
	python -m dtsr.bin.predict experiments_emnlp18\natstor.ini -a sampling -p train
	python -m dtsr.bin.predict experiments_emnlp18\natstor.ini -a sampling -p dev
	python -m dtsr.bin.predict experiments_emnlp18\natstor.ini -a sampling -p test
	python -m dtsr.bin.train experiments_emnlp18\dundee.ini
	python -m dtsr.bin.predict experiments_emnlp18\dundee.ini -a sampling -p train
	python -m dtsr.bin.predict experiments_emnlp18\dundee.ini -a sampling -p dev
	python -m dtsr.bin.predict experiments_emnlp18\dundee.ini -a sampling -p test
	python -m dtsr.bin.train experiments_emnlp18\ucl.ini
	python -m dtsr.bin.predict experiments_emnlp18\ucl.ini -a sampling -p train
	python -m dtsr.bin.predict experiments_emnlp18\ucl.ini -a sampling -p dev
	python -m dtsr.bin.predict experiments_emnlp18\ucl.ini -a sampling -p test
    python -m dtsr.bin.compare_to_baselines -t experiments_emnlp18\natstor.ini experiments_emnlp18\dundee.ini experiments_emnlp18\ucl.ini -p test

