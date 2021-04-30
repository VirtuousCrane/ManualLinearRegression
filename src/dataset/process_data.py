f = open("boston_train.csv");
train_line = [line for line in f.readlines()]
f.close()

f = open("train.csv" ,"w")
lbl = open("train_label.txt", "w")
for line in train_line[1:300]:
	f.write(",".join(line.split(",")[1:-1]))
	f.write("\n")
	lbl.write(line.split(",")[-1])
f.close()
lbl.close()

f = open("test.csv" ,"w")
lbl = open("test_label.txt", "w")
for line in train_line[300:]:
	f.write(",".join(line.split(",")[1:-1]))
	f.write("\n")
	lbl.write(line.split(",")[-1])
f.close()
lbl.close()
