f = open("kc_house_data.csv")
lines = [line for line in f.readlines()]
f.close();

prices = []
sqft_living = []

for line in lines:
	d = line.strip().split(",")
	prices.append(d[2])
	sqft_living.append(d[5])

p = int(len(prices) * 0.8)

f = open("train.txt", "w");
for i in prices[1:p]:
	f.write(str(float(i)) + "\n")
f.close()

f = open("train_lbl.txt", "w");
for i in sqft_living[1:p]:
	f.write(str(float(i)) + "\n")
f.close()

f = open("test.txt", "w");
for i in prices[p:]:
	f.write(str(float(i)) + "\n")
f.close()

f = open("test_lbl.txt", "w");
for i in sqft_living[p:]:
	f.write(str(float(i)) + "\n")
f.close()
