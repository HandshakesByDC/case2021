

data: data/20210312

data/20210312:
	curl -L 'https://www.dropbox.com/sh/i44patmfz2lbqmk/AAD8ixhwCxVoujU6OBgBe1fia?dl=1' -o casedata.zip
	unzip casedata.zip -x / -d data
	rm -rf casedata.zip

task4: subtask4/data

subtask4/data:
	cd subtask4 && mkdir -p data
	cp data/20210312/subtask4-token/en-train.txt subtask4/data/en-orig.txt
	cp data/20210312/subtask4-token/es-train.txt subtask4/data/es-orig.txt
	cp data/20210312/subtask4-token/pr-train.txt subtask4/data/pt-orig.txt
