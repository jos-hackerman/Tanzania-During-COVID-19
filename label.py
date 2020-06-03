

# def label(x):
# 	x1 = int(x)
# 	if x1 >= 0: return "1"
# 	else: return "0.0001"

f = open("graph_magafuli.txt", "r", encoding="utf-8")
f1 = open("graph_magafuli_label.csv", "w+", encoding="utf-8")

f1.write("source, target,weight" + "\n")
f_lines = f.readlines()

for line in f_lines[1:]:
	t = line.split(",")
	if t[2] != 0: 
		 f1.write(t[0] + "," + t[1] + "," + t[2] + "\n")

f.close()
f1.close()