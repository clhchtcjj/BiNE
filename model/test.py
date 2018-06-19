if __name__ == "__main__":
	with open(r"../data/mov/rating_train_r.dat",'r') as fr, open(r"../data/mov/rating_train.dat",'w') as fw:
		line = fr.readline()
		while line:
			items = line.split("\t")
			fw.write("u{}\ti{}\t{}\n".format(items[0],items[1],items[2]))
			line = fr.readline()

	with open(r"../data/mov/rating_test_r.dat",'r') as fr, open(r"../data/mov/rating_test.dat",'w') as fw:
		line = fr.readline()
		while line:
			items = line.split("\t")
			fw.write("u{}\ti{}\t{}\n".format(items[0],items[1],items[2]))
			line = fr.readline()

	
