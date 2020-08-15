res = ""
with open("README_template.rst", "r") as f:
    for line in f.readlines():
        line = line.strip("\n").replace(".. include:: ", "")
        if line.endswith(".rst"):
            tf = open(line, "r").read()
            res += tf.replace(".. literalinclude:: ../LICENSE", "").replace("<../LICENSE>", "<LICENSE>").replace(":class:", " ")
        else:
            res += line + "\n"

    print(res)

with open("README.rst", "w+") as f:
    f.write(res)
